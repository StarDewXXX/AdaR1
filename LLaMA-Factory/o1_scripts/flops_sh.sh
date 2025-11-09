# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/calculate_flops.py --model AIDC-AI/Marco-o1 --input_file Marco --mode normal
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/calculate_flops.py --model AIDC-AI/Marco-o1 --input_file Marco --mode very_fast
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/calculate_flops.py --model AIDC-AI/Marco-o1 --input_file Marco-dpo-iter0-beta-0.1 --mode normal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/calculate_flops.py --model AIDC-AI/Marco-o1 --input_file Marco-sft-shortest-iter0 --mode normal
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/calculate_flops.py --model AIDC-AI/Marco-o1 --input_file Marco-loft-iter0-alpha_2-lb_-1-ppo-normed --mode normal
