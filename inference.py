import argparse
import os
import time
from functools import partial

import numpy as np
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from ldm.models.diffusion.plms import PLMSSampler
from utils.checkpoint import load_model_ckpt

from utils.model import alpha_generator, set_alpha_scale, create_grounding_tokenizer, create_clip_pretrain_model, \
    get_clip_feature
from dataset.test_dataset import add_boxes, create_zero_input_tensors
import torch
device = 'cuda'


def prepare_batch(meta, batch=1, max_objs=30, model=None, processor=None, image_size=64, use_masked_att=False,
                  device="cuda"):

    phrases = meta.get("phrases")
    phrases = [None] * len(phrases) if phrases == None else phrases

    boxes, masks, text_masks, text_embeddings, _ = create_zero_input_tensors(max_objs)


    text_features = []
    for phrase in phrases:
        text_features.append(get_clip_feature(model, processor, phrase, is_image=False))

    for idx, (box, text_feature) in enumerate(zip(meta['locations'], text_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1

    out = {
        "boxes": boxes.unsqueeze(0).repeat(batch, 1, 1),
        "masks": masks.unsqueeze(0).repeat(batch, 1),
        "text_embeddings": text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
    }

    return out


@torch.no_grad()
def run(args,metas):
    # load models
    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(args.ckpt_path, args, device)

    grounding_tokenizer_input = create_grounding_tokenizer(config, model)

    # create dataloader
    clip_model, clip_processor = create_clip_pretrain_model()
    for meta in tqdm(metas):
        # prepare batch inputs
        # text prompt

        batch = prepare_batch(meta, args.num_images, args.max_objs, clip_model, clip_processor)

        context = text_encoder.encode([meta["prompt"]]*args.num_images )
        # negative prompt
        uc = text_encoder.encode( args.num_images*[""] )
        if args.negative_prompt is not None:
            uc = text_encoder.encode( args.num_images*[args.negative_prompt] )

        # sampler

        alpha_generator_func = partial(alpha_generator, type=[args.alpha, 0, 1.0 - args.alpha])


        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            # sampler = PLMSSampler(diffusion, model, set_alpha_scale=set_alpha_scale)

        # grounding input
        grounding_input = grounding_tokenizer_input.prepare(batch)
        # model inputs
        input = dict(x = None, timesteps = None, context = context, grounding_input = grounding_input)

        # model inputs for each instance if MIS is applied

        input_all = input

        # start sampling
        steps = 50
        shape = (args.num_images, model.in_channels, model.image_size, model.image_size)
        start_time = time.time()
        samples_fake = sampler.sample(S=steps, shape=shape, input=input_all,  uc=uc, guidance_scale=args.guidance_scale)
        samples_fake = autoencoder.decode(samples_fake)
        end_time = time.time()
        print(end_time-start_time)

        # folder for saving results
        output_folder = args.folder
        os.makedirs( output_folder, exist_ok=True)

        for idx, sample in enumerate(samples_fake):
            img_name = meta['file_name']
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255
            sample = Image.fromarray(sample.astype(np.uint8))
            if args.with_boxes:
                sample = add_boxes(sample, batch["boxes"][idx], batch["masks"][idx])
            sample.save(  os.path.join(output_folder, img_name[:-4]+f'_{idx}.png')   )
            print("image saved at: ", os.path.join(output_folder, img_name))

def get_args_parser():
    parser = argparse.ArgumentParser('Eval script', add_help=True)
    parser.add_argument("--folder", type=str,  default="inference_samples", help="root folder for output")
    parser.add_argument("--max_objs", type=int, default="10")
    parser.add_argument("--seed", type=int, default="123")
    parser.add_argument("--num_images", type=int, default=1, help="") # defalt=5
    parser.add_argument("--workers", type=int, default=0, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='cropped, worst quality, low quality', help="")
    parser.add_argument("--ckpt_path", type=str, default="D:\cocotrain2017\checkpoint_latest.pth", help="")
    parser.add_argument("--save_dir", type=str, default="output", help="")
    parser.add_argument("--alpha", type=float,  default=1, help="alpha for the percentage of timestep using grounded information")
    parser.add_argument("--with_boxes", type=bool, default=True)

    args = parser.parse_args()
    # return parser
    return args

if __name__ == '__main__':
    args = get_args_parser()
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    metas = [{
        'file_name': 'output1.png',
        'prompt': 'a person and a car in clear background.',
        'phrases': ['person','car'],
        'locations': [[0.25,0.35,0.45,0.80],[0.58,0.10,0.78,0.20]]
    }]
    run(args,metas)