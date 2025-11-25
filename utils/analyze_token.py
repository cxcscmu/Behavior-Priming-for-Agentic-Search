#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from token_calculator import tokenize

def extract_think_content(text):
    """extract content between <think> and </think>"""
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def analyze_jsonl_files(directory):
    """analyze all .jsonl files in the directory"""
    think_results = []
    input_results = []
    output_results = []
    total_files = 0
    total_think_blocks = 0
    total_input_blocks = 0
    total_output_blocks = 0
    total_think_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    # get all .jsonl files
    jsonl_files = list(Path(directory).glob('*.jsonl'))
    
    # add progress bar
    from tqdm import tqdm
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        total_files += 1
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            total_output_blocks += 1
                            output_tokens = tokenize(data['response'])
                            total_output_tokens += output_tokens
                            output_results.append({
                                'output_tokens': output_tokens
                            })
                            response = data['response']
                            
                            # extract think tag content
                            think_contents = extract_think_content(response)
                            
                            for think_content in think_contents:
                                total_think_blocks += 1
                                # use tokenize function to calculate token number
                                think_tokens = tokenize(think_content)  # empty string as input, only calculate token number of think content
                                total_think_tokens += think_tokens
                                
                                think_results.append({
                                    'think_tokens': think_tokens
                                })
                        if 'input' in data:
                            total_input_blocks += 1
                            input_tokens = tokenize(data['input'])
                            total_input_tokens += input_tokens
                            input_results.append({
                                'input_tokens': input_tokens
                            })
                    except json.JSONDecodeError as e:
                        print(f"   warning: line {line_num} JSON parsing error: {e}")
                        continue
                        
        except Exception as e:
            print(f"   error: cannot read file {jsonl_file}: {e}")
            continue
    
    return think_results, input_results, output_results

def main():
    directory = "logs/webwalkerqa/qwen3_1.7b_sft1"
    
    
    think_results, input_results, output_results = analyze_jsonl_files(directory)
    
    
    # sort by token number and display detailed results
    input_results.sort(key=lambda x: x['input_tokens'], reverse=True)
    think_results.sort(key=lambda x: x['think_tokens'], reverse=True)
    output_results.sort(key=lambda x: x['output_tokens'], reverse=True)

    
    # token number distribution statistics
    token_ranges = [
        (0, 50),
        (51, 200),
        (201, 500),
        (501, 1000),
        (1001, 2000),
        (2001, 3072),
        (3073, 4096),
        (4097, 10240),
        (10241, 12288),
        (12289, 16384),
        (16385, 20480),
        (20480, float('inf'))
    ]
    
    print(f"\ntoken distribution:")
    print("-" * 80)
    print(f"think content token distribution:")
    for start, end in token_ranges:
        if end == float('inf'):
            count = len([r for r in think_results if r['think_tokens'] > start])
            print(f"{start}+ tokens: {count} ")
        else:
            count = len([r for r in think_results if start <= r['think_tokens'] <= end])
            print(f"{start}-{end} tokens: {count} ")
    print("-" * 80)
    print(f"input content token distribution:")
    for start, end in token_ranges:
        if end == float('inf'):
            count = len([r for r in input_results if r['input_tokens'] > start])
            print(f"{start}+ tokens: {count} ")
        else:
            count = len([r for r in input_results if start <= r['input_tokens'] <= end])
            print(f"{start}-{end} tokens: {count} ")
    print("-" * 80)
    print(f"output content token distribution:")
    for start, end in token_ranges:
        if end == float('inf'):
            count = len([r for r in output_results if r['output_tokens'] > start])
            print(f"{start}+ tokens: {count} ")
        else:
            count = len([r for r in output_results if start <= r['output_tokens'] <= end])
            print(f"{start}-{end} tokens: {count} ")

if __name__ == "__main__":
    main()