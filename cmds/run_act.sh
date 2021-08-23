MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"

CUDA_VISIBLE_DEVICES=0 python scripts/image_train.py --data_dir /homes/55/yansong/action-to-motion/dataset $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
