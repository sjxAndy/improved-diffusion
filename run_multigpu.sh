CUDA_VISIBLE_DEVICES=2,3 mpiexec -n 2 python scripts/image_train.py --data_dir /home/lyh-cgy/sjx/data/cifar_train --image_size 64 --num_channels 128 --num_res_blocks 3 --class_cond True --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 16
