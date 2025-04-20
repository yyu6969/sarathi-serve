conda activate sarathi
module load cuda/12.4
cd /work/nvme/bdkz/yyu69/sarathi-serve/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false