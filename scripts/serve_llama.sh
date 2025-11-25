source ~/miniconda3/bin/activate verl-agent
MODEL_DIR=/data/group_data/cx_group/verl_agent_shared

SFT_CHECKPOINT=$MODEL_DIR/checkpoint/apm_sft_llama_3.2_3b_instruct
LLAMA_CHECKPOINT=$MODEL_DIR/Llama-3.2/Llama-3.2-3B-Instruct
RL_CHECKPOINT=$MODEL_DIR/checkpoint/mhqa_llama_sft_behavior_grpo/global_step_300/huggingface

vllm serve $RL_CHECKPOINT --port 1820 --data-parallel-size 2 --tensor-parallel-size 2 

