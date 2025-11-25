#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
import random

total_count = 0
max_count = 20000
suffix = ""

def filter_results(idx):
    source_dir = Path(f"results/afm/gemini_2.5_flash_{idx}")
    target_dir = Path(f"results/afm/gemini_2.5_flash_{suffix}")
    source_log_dir = Path(f"logs/afm/gemini_2.5_flash_{idx}")
    target_log_dir = Path(f"logs/afm/gemini_2.5_flash_{suffix}")
    input_file = Path("evaluation/short/afm/sft_full_qa.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_log_dir.mkdir(parents=True, exist_ok=True)
    
    filtered_ids = []

    step_count = 0
    
    # for all json files in source_dir
    for json_file in source_dir.glob("*.json"):
        
        for no_idx in no_list:
            if f"result_{no_idx}.json" in json_file.name:
                continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # check filter conditions
            score = data.get("score")
            turns = data.get("turns")
            search_count = data.get("search count", 0)
            summary_count = data.get("summary count", 0)
            positive_judge_result = data.get("positive_judge_result", None)

            if positive_judge_result is None:
                continue

            behavior1 = positive_judge_result.get("behavior1", None)
            behavior2 = positive_judge_result.get("behavior2", None)
            behavior3 = positive_judge_result.get("behavior3", None)
            behavior4 = positive_judge_result.get("behavior4", None)

            all_positive = behavior1 == "Yes" and behavior2 == "Yes" and behavior3 == "Yes" and behavior4 == "Yes"

            condition = all_positive
            
            if condition:
                step_count += turns
                global total_count
                total_count += 1
                if total_count >= max_count:
                    return True
                
                # extract file ID (remove result_ prefix and .json suffix)
                file_id = json_file.stem.replace("result_", "")
                filtered_ids.append(file_id)
                
                ### copy result file to target directory
                target_file = target_dir / json_file.name
                if not target_file.exists():
                    shutil.copy2(json_file, target_file)
                    print(f"copied file: {json_file.name}")
                else:
                    for i in range(1, 10):
                        target_file = target_dir / f"{json_file.name.replace('.json', f'_{i}.json')}"
                        if not target_file.exists():
                            shutil.copy2(json_file, target_file)
                            print(f"copied file: {json_file.name}")
                            break
                
                ### copy jsonl file to target directory
                jsonl_file = source_log_dir / f"trajectory_{file_id}.jsonl"
                target_jsonl_file = target_log_dir / jsonl_file.name
                if not target_jsonl_file.exists():
                    shutil.copy2(jsonl_file, target_jsonl_file)
                    print(f"copied jsonl file: {jsonl_file.name}")
                else:
                    for i in range(1, 10):
                        target_jsonl_file = target_log_dir / f"{jsonl_file.name.replace('.jsonl', f'_{i}.jsonl')}"
                        if not target_jsonl_file.exists():
                            shutil.copy2(jsonl_file, target_jsonl_file)
                            print(f"copied jsonl file: {jsonl_file.name}")
                            break
                
        except Exception as e:
            print(f"error processing file {json_file.name}: {e}")
           
    
    filtered_data = []
    for item in all_data:
        if str(item['id']) in filtered_ids:
            filtered_data.append(item)
    
    print(f"\nfiltering completed!")
    print(f"files copied to: {target_dir}")
    print(f"number of filtered data: {len(filtered_data)}")
    return len(filtered_data) >= max_count

if __name__ == "__main__":
    for idx in range(1, 10):
        finished = filter_results(idx) 
        if finished:
            break
    print(f"total trajectory count: {total_count}")