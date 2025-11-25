from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed.tensor  
from glob import glob
from collections import defaultdict
from tqdm import tqdm

fsdp_checkpoint_path = "/data/group_data/cx_group/verl_agent_shared/checkpoint/llama_3.2_3b_sft_grpo/global_step_240/actor"
huggingface_model_path = "/data/group_data/cx_group/verl_agent_shared/Llama-3.2/Llama-3.2-3B-Instruct"
output_path = fsdp_checkpoint_path.replace("actor", "huggingface")

def main():
    state_dict = defaultdict(list)

    world_size = 8  # modify to match your checkpoint format
    for rank in tqdm(range(world_size), desc="Loading checkpoints"):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()