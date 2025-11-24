export CUDA_VISIBLE_DEVICES=0,1  # adjust count
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
accelerate launch --num_processes 2 simple_ddp.py