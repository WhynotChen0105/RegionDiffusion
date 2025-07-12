export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=8 main.py  --yaml_file=configs/voc_train_RSA.yaml  --batch_size=8 --workers 8 --name voc0712 --total_epochs 80 --disable_inference_in_training True
accelerate launch test.py --folder generations/crowdhuman --ann_file DATA/crowdhuman_train.json --ckpt_path OUTPUT/crowdhuman/tag02/checkpoint_latest.pth --min_area 0.02 --iou_threshold 0.3 --batch_size 12 --times 2
accelerate launch test.py --folder generations/cityscapes --ann_file DATA/cityscapes_train.json --ckpt_path OUTPUT/cityscapes/tag01/checkpoint_latest.pth --min_area 0.02 --iou_threshold 0.3 --batch_size 12 --times 2
python -m torch.distributed.launch --nproc_per_node=8 main.py  --yaml_file=configs/enginer_train_RSA.yaml  --batch_size=6 --workers 8 --name enginer_cars --total_epochs 80 --disable_inference_in_training True
accelerate launch test.py --folder generations/enginer_cars --ann_file DATA/enginer_train.json --ckpt_path OUTPUT/enginer_cars/tag00/checkpoint_latest.pth --min_area 0.02 --iou_threshold 0.3 --batch_size 12 --times 2
python -m torch.distributed.launch --nproc_per_node=8 main.py  --yaml_file=configs/sim10k_train_RSA.yaml  --batch_size=6 --workers 8 --name sim10k --total_epochs 80 --disable_inference_in_training True
accelerate launch test.py --folder generations/sim10k --ann_file DATA/sim10k_train.json --ckpt_path OUTPUT/sim10k/tag00/checkpoint_latest.pth --min_area 0.02 --iou_threshold 0.3 --batch_size 12 --times 2
top
