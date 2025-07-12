import math
from random import random
import torch.utils.data
from PIL import ImageDraw
import random
from pycocotools.coco import COCO
from tqdm import tqdm
from dataset.tsv_dataset import make_a_sentence
from utils.dist import get_rank
from utils.model import get_clip_feature, create_clip_pretrain_model
import torch

def add_boxes(img, boxes, masks):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for bid, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        if masks[bid] == 1:
            draw.rectangle([float(x0 * W), float(y0 * H), float(x1 * W), float(y1 * H)], outline = colors[bid % len(colors)], width = 2)
    return img

def transform(annos, image_info, random_crop=True, random_flip=True, min_crop_frac=1.0, max_crop_frac=1.0, min_area=0.00):
    new_annos = []
    if random_crop:
        width, hegiht = image_info['width'], image_info['height']
        length = min(width, hegiht)
        min_length = math.ceil(length * min_crop_frac)
        max_length = math.ceil(length * max_crop_frac)
        smaller_dim_size = random.randrange(min_length, max_length + 1)
        crop_y = random.randrange(hegiht - smaller_dim_size + 1)
        crop_x = random.randrange(width - smaller_dim_size +1)
        for anno in annos:
            x, y, w, h = anno['bbox']
            x0 = x - crop_x
            y0 = y - crop_y
            x1 = x0 + w
            y1 = y0 + h
            if x1 >=0 and y1 >=0 and x0 <= smaller_dim_size and y0 <= smaller_dim_size:
                x0 = max(0,x0) / smaller_dim_size
                y0 = max(0,y0) / smaller_dim_size
                x1 = min(smaller_dim_size, x1) / smaller_dim_size
                y1 = min(smaller_dim_size, y1) / smaller_dim_size
                if random_flip and random.random() < 0.5:
                    x0, x1 = 1 - x1, 1 - x0
                anno['bbox'] = x0, y0, x1, y1
                if (x1-x0) * (y1 - y0) >= min_area:
                    new_annos.append(anno)
    else: # letterbox
        for anno in annos:
            width, height = image_info['width'], image_info['height']
            x, y, w, h = anno['bbox']
            x0 = x / width
            y0 = y / height
            x1 = (x + w) / width
            y1 = (y + h) / height
            if random_flip and random.random() < 0.5:
                    x0, x1 = 1 - x1, 1 - x0
            anno['bbox'] = x0, y0, x1, y1
            if (x1 - x0) * (y1 - y0) >= min_area:
                new_annos.append(anno)

    return new_annos

#计算某个box与其他多个box的iou值
def iou(anno,annos, image_size=512):
    '''
    :param box: 矩形框box的坐标
    :param boxes: 一大堆矩形框的坐标
    :return: 计算矩形框box与一大堆矩形框的iou值
    '''
    # box.shape: (4,),4个值分别是x1,y1,x2,y2
    # boxes.shape: (n,4) ,n是要计算iou的矩形框个数

    box = torch.tensor(anno["bbox"])*image_size
    boxes = torch.zeros([len(annos),4])
    for i in range(len(annos)):
        boxes[i,:] = torch.tensor(annos[i]["bbox"])*image_size

    box_area= (box[2]-box[0]) * (box[3]-box[1]) #box的面积
    boxes_area= (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1]) #多个box的面积

    # 交集坐标
    xx1=torch.maximum(box[0],boxes[:,0]) #交集矩形框角坐标x1
    yy1=torch.maximum(box[1],boxes[:,1]) #交集矩形框角坐标y1
    xx2 = torch.minimum(box[2], boxes[:,2])  # 交集矩形框角坐标x2
    yy2 = torch.minimum(box[3], boxes[:,3])  # 交集矩形框角坐标y2

    # 交集面积
    inter_area=torch.max(torch.tensor((0)),xx2-xx1+1)*torch.max(torch.tensor((0)),yy2-yy1+1)

    #并集面积
    union_area=box_area + boxes_area - inter_area

    #返回iou
    iou=inter_area/union_area

    return iou


def nms(annos, thresh=0.3):

    new_annos = annos

    annos_result = []
    while len(new_annos) > 1:
        anno = new_annos[0]
        annos_result.append(anno)
        new_annos = new_annos[1:]
        annos_temp = []
        ious = iou(anno, new_annos)
        for i in range(len(ious)):
            if ious[i] < thresh:
                annos_temp.append(new_annos[i])
        new_annos = annos_temp

    if len(new_annos) > 0:
        annos_result.append(new_annos[0])

    return annos_result

def sort_by_area(annos):
    annos.sort(reverse=True, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))



def create_zero_input_tensors(max_objs):
    masks = torch.zeros(max_objs+1) # binay, indicates the instance conditioning exists or not
    text_masks = torch.zeros(max_objs+1) # binay, indicates the instance conditioning exists or not
    text_embeddings = torch.zeros(max_objs+1, 768)
    boxes_embeddings = torch.zeros(max_objs+1, 4)
    categorys = torch.zeros(max_objs+1)
    return boxes_embeddings, masks, text_masks, text_embeddings, categorys


def add_boxes(img, boxes, masks):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for bid, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        if masks[bid] == 1:
            draw.rectangle([float(x0 * W), float(y0 * H), float(x1 * W), float(y1 * H)], outline = colors[bid % len(colors)], width = 2)
    return img

# convert boxes' coordinates to the relative values (0, 1)
def convert_coco_box(bbox, img_info):
    length = max(img_info['width'], img_info['height'])
    x0 = bbox[0] / length
    y0 = bbox[1] / length
    x1 = (bbox[0] + bbox[2] ) / length
    y1 = (bbox[1] + bbox[3] ) / length

    return [x0, y0, x1, y1]



def prepare(meta, max_objs=30, model=None, processor=None):
    file_name = meta.get('file_name')

    prompt = meta.get('prompt')

    phrases = meta.get("phrases")

    cat_ids = meta.get("cat_ids")

    boxes, masks, text_masks, text_embeddings, categories = create_zero_input_tensors(max_objs)

    text_features = []
    for phrase in phrases:
        text_features.append(get_clip_feature(model, processor, phrase, is_image=False))

    for idx, (box, text_feature) in enumerate(zip(meta['locations'], text_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        categories[idx] = torch.tensor(cat_ids[idx])
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1

    out = {
        "cat_ids": categories,
        "file_name": file_name,
        "prompt": prompt,
        "boxes": boxes,
        "masks": masks,
        "text_embeddings": text_embeddings
    }

    return out


class inference_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        max_objs = args.max_objs
        self.max_objs = max_objs
        # clip
        self.clip_model, self.clip_processor = create_clip_pretrain_model()
        # read MSCOCO
        ann_file = args.instances_file
        coco = COCO(ann_file)
        image_ids = coco.getImgIds()
        categories = coco.cats
        id_map = {} # for yolo, remapping
        for idx, cat_id in enumerate(categories.keys()):
            id_map[cat_id] = idx
        image_ids.sort()
        # start image generation
        self.meta_dict_list = []
        if args.use_captions:
            coco_caption = COCO(args.captions_file)
        # print("preparing the dataset...")
        for img_id in tqdm(image_ids ,disable = get_rank() != 0):
            test_info = dict(
                prompt=None,
                phrases=None,
                locations=None,
                file_name=None,
            )
            # Pick one image.
            img_info = coco.loadImgs([img_id])[0]
            test_info['file_name'] = img_info['file_name']
            # Get all the annotations for the specified image.
            try: #for coco
                ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            except: # for lvis
                ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            if args.sort_by_area:
                sort_by_area(anns)
            anns = transform(anns, img_info ,random_crop=args.random_crop, random_flip=args.random_flip, min_area=args.min_area)
            anns = nms(anns,thresh=args.iou_threshold)
            # get [x1,y1,x2,y2] format locations
            test_info['locations'] = [ann["bbox"] for ann in anns][:max_objs]
            # get categories
            cat_ids = [ann['category_id'] for ann in anns]
            cats = coco.loadCats(cat_ids)
            cat_ids = [id_map[cat_id] for cat_id in cat_ids]
            cat_names = [cat["name"].replace("_"," ") for cat in cats]
            test_info['phrases'] = cat_names[:max_objs]
            if args.use_captions:
                ann_ids_captions = coco_caption.getAnnIds(imgIds=[img_id], iscrowd=None)
                caption = coco_caption.loadAnns(ann_ids_captions)[0]['caption']
                caption = make_a_sentence(cat_names, scene=args.scene, caption = caption, concat= args.concat)
            else:
                caption = make_a_sentence(cat_names, scene=args.scene)
            test_info['prompt'] = caption
            test_info['cat_ids'] = cat_ids[:max_objs]
            self.meta_dict_list.append(test_info)
        # print(self.meta_dict_list)
    def __getitem__(self, item):
        return prepare(self.meta_dict_list[item],max_objs=self.max_objs, model=self.clip_model, processor=self.clip_processor)

    def __len__(self):
        return len(self.meta_dict_list)