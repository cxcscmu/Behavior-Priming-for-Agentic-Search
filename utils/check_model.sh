dataset=("webwalkerqa" "hle" "gaia")
suffix="random_sft"
# checkpoint_path="/data/group_data/cx_group/verl_agent_shared/checkpoint/apm_1.7b_sft_incorrect_and_positive_grpo/global_step_300/huggingface"
checkpoint_path="/data/group_data/cx_group/verl_agent_shared/checkpoint/apm_sft_1.7b_random"



# # for checking model
for dataset in ${dataset[@]}; do
    python utils/check_model.py --results_dir results/${dataset}/${suffix} --logs_dir logs/${dataset}/${suffix} --target_model_path ${checkpoint_path}
done

# # for deleting model, if you want to delete the model, uncomment the following code
# for dataset in ${dataset[@]}; do
#     python utils/check_model.py --results_dir results/${dataset}/${suffix} --logs_dir logs/${dataset}/${suffix} --target_model_path ${checkpoint_path} --execute
# done

