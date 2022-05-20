MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --schedule_sampler loss-second-moment"

CUDA_VISIBLE_DEVICES=0,1,2 OPENAI_LOGDIR=/home/sjx22/openai_logs/openai_logs13 mpiexec -n 3 python scripts/image_train.py --data_dir /home/sjx22/data_proc/6 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS