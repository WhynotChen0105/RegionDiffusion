import torch
import json
from torch.utils.data import Dataset
import torchvision
from PIL import ImageDraw
import random
from .utils import TSVFile
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from .utils import random_crop_arr, center_crop_arr, recalculate_box_and_verify_if_valid


def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    # new_arr = np.empty_like(arr)
    new_arr = np.copy(arr)
    if use_tensor:
        arr = torch.from_numpy(new_arr)
    return arr

def decode_item(item):
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        # anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        # anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item

def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field

def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)
        data_info.pop("data_source", None)
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None) # I have checked that all 0 for flickr, vg, coco
        anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
        anno_info.pop("area", None)
        # anno_info.pop("id", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1
    return img


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


def make_a_sentence(obj_names, scene = None, caption = None, concat=False):

    if len(obj_names) == 0 or obj_names is None:

        prompt = 'An image of nothing.'
    else:
        instances_prompt = ''

        scene = f' in {scene} scene' if scene is not None else ''
        for obj_name in obj_names:
            instances_prompt = instances_prompt +  obj_name + ' and '
        instances_prompt = instances_prompt[:-5] # remove the ' and '

        prompt = f'An image of {instances_prompt}{scene}.'

    if caption is not None:
        if concat == True:
            return prompt[:-1] + ' where ' + caption
        else:
            return caption
    #
    # if caption is not None:
    #
    #     prompt = f'An image of {instances_prompt}{scene}, where {caption}'

    return prompt #, tokens_positive


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token,
    but we always make sure there is at least one feature used.
    In other words, the following masks are not valid
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature
    and always keep the text one.

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature
                temp_mask[idx,i] = 0
        # image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks

    if random_drop_embedding=='image':
        # image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    # return image_masks, text_masks

    return  text_masks




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim).
    this function will return the CLIP penultimate feature.

    Note: to make sure getting the correct penultimate feature, the input y should not be normalized.
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)




class TSVDataset(Dataset):
    def __init__(self,
                tsv_path,
                which_layer_text='before',
                prob_use_caption=1,
                random_drop_embedding='none',
                image_size=512,
                min_box_size=0.04,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                loss_size = 64,
                foreground_loss_weight = 2,
                foreground_loss_norm = True,
                scene = None,
                concat = False,
                ):
        super().__init__()
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.image_size = image_size
        self.tsv_path = tsv_path
        self.which_layer_text  = which_layer_text
        # self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        self.loss_size = loss_size
        self.foreground_loss_weight = foreground_loss_weight
        self.foreground_loss_norm = foreground_loss_norm
        self.scene = scene
        self.concat = concat
        assert which_layer_text in ['before','after']
        # assert which_layer_image in ['after', 'after_renorm', 'after_reproject']
        assert random_drop_embedding in ['none', 'both', 'image']


        # Last linear layer used in CLIP text encoder. Here we use it to map CLIP image embedding into penultimate text space. See Appendix in paper.
        # self.projection_matrix = torch.load('projection_matrix')

        # Load tsv data
        self.tsv_file = TSVFile(self.tsv_path)

        # preprocessed CLIP feature embedding length: 768
        self.embedding_len = 768


    def total_images(self):
        return len(self)

    def get_item_from_tsv(self, index):
        _, item = self.tsv_file[index]
        item = decode_item(item)
        return item



    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        raw_item = self.get_item_from_tsv(index)
        is_det = raw_item.get('is_det', False) # if it is from detection (such as o365), then we will make a pseudo caption

        out = {}

        # -------------------- id and image ------------------- #
        out['id'] = raw_item['data_id']
        image = raw_item['image']
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor

        # -------------------- grounding token ------------------- #
        annos = raw_item['annos']
        captions = raw_item['captions']

        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_category_names = []
        text_embedding_name = 'text_embedding_before' if self.which_layer_text == 'before' else 'text_embedding_after'

        for anno in annos:
            x, y, w, h = anno['bbox']
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)
            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                all_text_embeddings.append(anno[text_embedding_name])
                all_category_names.append(anno["category_name"])
        # Sort according to area and choose the largest N objects
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1].tolist()
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]
        random.shuffle(wanted_idxs)
        boxes = torch.zeros(self.max_boxes_per_data+1, 4) # instances and background
        masks = torch.zeros(self.max_boxes_per_data+1) # instances and background
        loss_mask = torch.zeros([self.loss_size,self.loss_size])
        text_embeddings =  torch.zeros(self.max_boxes_per_data+1, self.embedding_len) # instances and background
        if is_det:
            category_names = []
        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            text_embeddings[i] =  all_text_embeddings[idx]
            category_names.append(all_category_names[idx])

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            # image_masks = masks
            text_masks = masks
        for box, mask in zip(boxes,masks):
            if mask == 1:
                coord = torch.round(box * self.loss_size).int().tolist()
                loss_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(
                    torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),0.2)
        loss_mask[loss_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.loss_size * self.loss_size), 0.2)
        loss_mask = loss_mask / torch.sum(loss_mask) * self.loss_size * self.loss_size if self.foreground_loss_norm else loss_mask
        loss_mask = loss_mask.unsqueeze(0)
        out["loss_masks"] = loss_mask
        out["boxes"] = boxes
        out["masks"] = masks # indicating how many valid objects for this image-text data
        out["text_masks"] = text_masks # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings


        # -------------------- caption ------------------- #
        if random.uniform(0, 1) < self.prob_use_caption:
            caption = random.sample(captions, 1)[0]
            out["caption"] = make_a_sentence(category_names, self.scene, caption ,self.concat)
        else:
            out["caption"] = make_a_sentence(category_names, self.scene)
        return out



    def __len__(self):
        if self.max_images is None:
            return len(self.tsv_file)
        return min(len(self.tsv_file), self.max_images)


    def vis_getitem_data(self, index=None, out=None, sample=None, return_tensor=False, name="res.jpg",
                         print_caption=True):

        if out is None:
            out = self[index]

        img = torchvision.transforms.functional.to_pil_image(out["image"] * 0.5 + 0.5)
        # canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        sample = torchvision.transforms.functional.to_pil_image(sample * 0.5 + 0.5)
        W, H = img.size

        if print_caption:
            caption = out["caption"]
            print(caption)
            print(" ")

        boxes = []
        for box, mask in zip(out["boxes"], out["masks"]):
            if mask == 1:
                x0, y0, x1, y1 = box
                boxes.append([float(x0 * W), float(y0 * H), float(x1 * W), float(y1 * H)])
        img = draw_box(img, boxes)
        sample = draw_box(sample, boxes)
        if return_tensor:
            return torchvision.transforms.functional.to_tensor(img), torchvision.transforms.functional.to_tensor(sample)
        else:
            img.save(name)

    def transform_image(self, pil_image):
        if self.random_crop:
            arr, info = random_crop_arr(pil_image, self.image_size)
        else:
            arr, info = center_crop_arr(pil_image, self.image_size)

        info["performed_flip"] = False
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True

        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])

        return torch.tensor(arr), info