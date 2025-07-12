export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
# captions
accelerate launch ./main.py --name caption_v1 --yaml_file configs/coco10k_captions_v1.yaml --warmup_steps 500 --batch_size 8 --workers 4 --total_epochs 40 --save_every_iters 3000 --disable_inference_in_training
accelerate launch test.py --folder generations/captions_v1  --ckpt_path OUTPUT/caption_v1/checkpoint_latest.pth --batch_size 10 --times 1 --workers 4 --use_captions
# classnames
accelerate launch ./main.py --name caption_v2 --yaml_file configs/coco10k_captions_v2.yaml --warmup_steps 500 --batch_size 8 --workers 4 --total_epochs 40 --save_every_iters 3000 --disable_inference_in_training
accelerate launch test.py --folder generations/captions_v2  --ckpt_path OUTPUT/caption_v2/checkpoint_latest.pth --batch_size 10 --times 1 --workers 4
# concat
accelerate launch ./main.py --name caption_v3 --yaml_file configs/coco10k_captions_v3.yaml --warmup_steps 500 --batch_size 8 --workers 4 --total_epochs 40 --save_every_iters 3000 --disable_inference_in_training
accelerate launch test.py --folder generations/captions_v3  --ckpt_path OUTPUT/caption_v3/checkpoint_latest.pth --batch_size 10 --times 1 --workers 4 --use_captions --concat

top
