source ~/miniconda3/bin/activate verl-agent

if [ $# -lt 2 ]; then
    echo "Usage: $0 <suffix> <port>"
    exit 1
fi

suffix=$1
port=$2
url=localhost:$port
echo "Running $suffix"

for idx in {0..3}; do
    echo "Running on $idx times for $suffix"

    echo "Running on multi-hop qa"
    python3 main_parallel.py --batch_file evaluation/short/mhqa/test_small_deepresearcher.json \
        --answer_dir results/mhqa/$suffix \
        --log_dir logs/mhqa/$suffix \
        --is_llama \
        --url http://$url/v1 \
        --use_explicit_thinking \
        --search_engine serper \
        --max_turns 15

    echo "Running on webwalkerqa"
    python3 -u main_parallel.py --batch_file evaluation/short/webwalkerqa/test.json \
        --answer_dir results/webwalkerqa/$suffix \
        --log_dir logs/webwalkerqa/$suffix \
        --is_llama \
        --url http://$url/v1 \
        --use_explicit_thinking \
        --search_engine serper \
        --max_turns 25

    echo "Running on gaia"
    python3 -u main_parallel.py --batch_file evaluation/short/gaia/test.json \
        --answer_dir results/gaia/$suffix \
        --log_dir logs/gaia/$suffix \
        --is_llama \
        --url http://$url/v1 \
        --use_explicit_thinking \
        --search_engine serper \
        --max_turns 25

    echo "Running on hle"
    python3 -u main_parallel.py --batch_file evaluation/short/hle/test.json \
        --answer_dir results/hle/$suffix \
        --log_dir logs/hle/$suffix \
        --is_llama \
        --url http://$url/v1 \
        --use_explicit_thinking \
        --search_engine serper \
        --max_turns 25
done

echo "Finished $suffix!"