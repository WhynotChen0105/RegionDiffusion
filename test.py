import argparse
import os
from functools import partial
import numpy as np
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm
from ldm.models.diffusion.plms import PLMSSampler
from utils.checkpoint import load_model_ckpt
from utils.model import alpha_generator, set_alpha_scale
from dataset.test_dataset import inference_dataset, add_boxes
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

def write_into_yolo(boxes, cat_ids, masks, file_name):
    with open(file_name, mode='w') as f:
        for i in range(len(masks)):
            if masks[i] == 1:
                x1, y1, x2, y2 = boxes[i]
                x = float((x1 + x2) / 2.0)
                y = float((y1 + y2) / 2.0)
                w = float(x2-x1)
                h = float(y2-y1)
                f.write(f"{int(cat_ids[i])} {x} {y} {w} {h}\n")

@torch.no_grad()
def run(args):
    accelerate = Accelerator()
    # load models
    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(args.ckpt_path, args, accelerate.device)
    grounding_tokenizer_input = model.grounding_tokenizer_input
    model, autoencoder, text_encoder, diffusion = accelerate.prepare(model, autoencoder, text_encoder, diffusion)

    # create dataloader

    if accelerate.use_distributed:
        model = model.module
        autoencoder = autoencoder.module
    for time in range(args.times):
        # set the seed
        seed = args.seed + time
        set_seed(seed)
        dataset = inference_dataset(args)
        # assert len(dataset) % args.batch_size == 0
        dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, drop_last=True)
        dataloader = accelerate.prepare(dataloader)

        for batch in tqdm(dataloader, disable = not accelerate.is_main_process):
            # prepare batch inputs
            # text prompt
            # batch_to_device(batch,device)
            print(batch["prompt"])
            context = text_encoder.encode(batch["prompt"])
            # negative prompt
            uc = text_encoder.encode( args.batch_size*[""] )
            if args.negative_prompt is not None:
                uc = text_encoder.encode( args.batch_size*[args.negative_prompt] )

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

            shape = (args.batch_size, model.in_channels, model.image_size, model.image_size)
            samples_fake = sampler.sample(S=steps, shape=shape, input=input_all,  uc=uc, guidance_scale=args.guidance_scale)
            samples_fake = autoencoder.decode(samples_fake)

            # folder for saving results
            output_folder = args.folder
            # os.makedirs( output_folder, exist_ok=True)
            os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, 'bbox'), exist_ok=True)
            os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
            for idx, sample in enumerate(samples_fake):
                img_name = batch['file_name'][idx]
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = sample.cpu().numpy().transpose(1,2,0) * 255
                sample = Image.fromarray(sample.astype(np.uint8))
                if args.rename:
                    img_name = 'fake_' + img_name.replace("\\", "_")[:-4] + f'_{time}.png'
                sample.save(os.path.join(output_folder, 'images', img_name))
                if args.with_boxes:
                    sample = add_boxes(sample, batch["boxes"][idx], batch["masks"][idx])
                    sample.save(os.path.join(output_folder, 'bbox', img_name))
                    write_into_yolo(batch["boxes"][idx], batch["cat_ids"][idx], batch["masks"][idx], os.path.join(output_folder, "labels", img_name[:-4]+'.txt'))
                print("image saved at: ", os.path.join(output_folder, img_name))


def get_args_parser():
    parser = argparse.ArgumentParser('Test script', add_help=True)
    parser.add_argument("--folder", type=str,  default="generation_samples/cocoval17", help="root folder for output")
    parser.add_argument("--instances_file", type=str, default="DATA/instances_val2017-4k.json")
    parser.add_argument("--captions_file", type=str, default="DATA/captions_val2017.json")
    parser.add_argument("--max_objs", type=int, default=30)
    parser.add_argument("--seed", type=int, default="123")
    parser.add_argument("--batch_size", type=int, default=5, help="") # defalt=5
    parser.add_argument("--workers", type=int, default=0, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--ckpt_path", type=str, default="", help="")
    parser.add_argument("--alpha", type=float,  default=1.0, help="alpha for the percentage of timestep using grounded information")
    parser.add_argument("--with_boxes", action='store_true')
    parser.add_argument("--min_area", type=float, default=0.00)
    parser.add_argument("--iou_threshold", type=float, default=1.0)
    parser.add_argument("--sort_by_area", action='store_true')
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--rename", action='store_true')
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--random_crop", action='store_true')
    parser.add_argument("--random_flip", action='store_true')
    parser.add_argument("--use_captions", action='store_true')
    parser.add_argument("--concat", action='store_true')
    args = parser.parse_args()
    # return parser
    return args

if __name__ == '__main__':
    args = get_args_parser()
    cudnn.benchmark = True
    run(args)