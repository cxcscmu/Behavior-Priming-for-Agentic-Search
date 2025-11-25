source ~/miniconda3/bin/activate verl-agent

suffix=qwen3_1.7b_sft_behavior

echo "Running on mhqa"
python evaluation/short/evaluation.py --data_path evaluation/short/mhqa/test_small_deepresearcher.json --results_dir results/mhqa/$suffix --mhqa --enable_hack_detection
echo "--------------------------------"

echo "Running on webwalkerqa"
python evaluation/short/evaluation.py --data_path evaluation/short/webwalkerqa/test.json --results_dir results/webwalkerqa/$suffix --enable_hack_detection
echo "--------------------------------"

echo "Running on hle"
python evaluation/short/evaluation.py --data_path evaluation/short/hle/test.json --results_dir results/hle/$suffix --enable_hack_detection
echo "--------------------------------"

echo "Running on gaia"
python evaluation/short/evaluation.py --data_path evaluation/short/gaia/test.json --results_dir results/gaia/$suffix --enable_hack_detection
echo "--------------------------------"


echo "Finished $suffix!"
