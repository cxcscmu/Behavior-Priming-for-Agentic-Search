#!/bin/bash

source ~/miniconda3/bin/activate verl-agent

ANALYSIS_SCRIPT="behaviour_analysis/analysis.py"


suffix_list=("gemini_2.5_flash")
for suffix in ${suffix_list[@]}; do
    echo "Analyzing trajectories for suffix: $suffix"
    
    python $ANALYSIS_SCRIPT \
        --logs_dir "logs/afm_mhqa/${suffix}" \
        --results_dir "results/afm_mhqa/${suffix}" \
        --question_path "evaluation/short/afm_mhqa/sft_full_qa.json" 
    echo "--------------------------------"

    echo "Running on webwalkerqa"
    python $ANALYSIS_SCRIPT \
        --logs_dir "logs/webwalkerqa/best_of_n/${suffix}" \
        --results_dir "results/webwalkerqa/best_of_n/${suffix}" \
        --question_path "evaluation/short/webwalkerqa/test_random_250.json" 
    echo "--------------------------------"

    echo "Running on hle"
    python $ANALYSIS_SCRIPT \
        --logs_dir "logs/hle/${suffix}" \
        --results_dir "results/hle/${suffix}" \
        --question_path "evaluation/short/hle/test.json" 
    echo "--------------------------------"

    echo "Running on gaia"
    python $ANALYSIS_SCRIPT \
        --logs_dir "logs/gaia/${suffix}" \
        --results_dir "results/gaia/${suffix}" \
        --question_path "evaluation/short/gaia/test.json" 
    echo "--------------------------------"


echo "Finished analyzing for suffix: $suffix!"
done

