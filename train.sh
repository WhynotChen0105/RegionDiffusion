export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch ./main.py --name cocotrain2017 --yaml_file configs/cocotrain2017.yaml --warmup_steps 5000 --batch_size 4 --workers 4 --total_epochs 80 --save_every_iters 80000 --accumulation_steps 4
accelerate launch test.py --folder generations/cocoval17  --ckpt_path OUTPUT/cocotrain2017/checkpoint_latest.pth --batch_size 10 --times 1 --workers 4 --use_captions --concat

top
