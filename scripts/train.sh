export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)


if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
else
    echo "Detected $GPU_MODEL, keeping NCCL P2P enabled"
fi

source ~/miniconda3/bin/activate llama_factory

cd train/sft/LLaMA-Factory 

mkdir -p train/sft/log

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train/sft/log/training_${TIMESTAMP}.log"



echo "start training" >> "$LOG_FILE"
llamafactory-cli train ../sft.yaml 2>&1 | tee "$LOG_FILE" 
