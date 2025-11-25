source ~/miniconda3/bin/activate qwen_agent

MODEL_DIR=/data/group_data/cx_group/verl_agent_shared

RL_CHECKPOINT=$MODEL_DIR/checkpoint/webwalker_1.7b_sft_grpo_sparse_reward_3/global_step_250/huggingface
vllm serve $RL_CHECKPOINT --port 8000 --data-parallel-size 1 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072

