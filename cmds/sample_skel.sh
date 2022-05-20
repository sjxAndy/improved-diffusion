MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"

CUDA_VISIBLE_DEVICES=2 OPENAI_LOGDIR=/home/sjx22/openai_logs/openai_sample_logs_13_test python scripts/image_sample.py --model_path /home/sjx22/openai_logs/openai_logs13/ema_0.9999_020000.pt $MODEL_FLAGS $DIFFUSION_FLAGS
