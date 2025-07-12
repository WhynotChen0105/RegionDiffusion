import argparse
import accelerate.utils
from omegaconf import OmegaConf
from trainer import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str, default="DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str, default="OUTPUT", help="path to OUTPUT")
    parser.add_argument("--name", type=str, default="RSA", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int, default=123, help="used in sampler")
    parser.add_argument("--yaml_file", type=str, default="configs/coco2017train_RSA_64_32_16_captions.yaml", help="paths to base configs.")
    parser.add_argument("--base_learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="")
    parser.add_argument("--scheduler_type", type=str, default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation steps")
    parser.add_argument("--workers", type=int, default=2, help="")
    parser.add_argument("--official_ckpt_name", type=str, default="pretrained/v1-5-pruned-emaonly.ckpt",
                        help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument('--enable_ema', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="")
    parser.add_argument("--total_iters", type=int, default=500000, help="")
    parser.add_argument("--total_epochs", type=int, default=80, help="")
    parser.add_argument("--save_every_iters", type=int, default=20000, help="")
    parser.add_argument("--disable_inference_in_training", action='store_true',
                        help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")
    parser.add_argument('--foreground_loss_mode',default=True)

    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']

    config = OmegaConf.load(args.yaml_file)

    config.update(vars(args))
    accelerate.utils.set_seed(config.seed)
    trainer = Trainer(config)
    trainer.start_training()

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py  --yaml_file=configs/ade_sem.yaml  --DATA_ROOT=../../DATA   --batch_size=4