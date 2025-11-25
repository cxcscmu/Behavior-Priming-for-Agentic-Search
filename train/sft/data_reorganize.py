#!/usr/bin/env python3
"""
Data reorganization script: Convert .jsonl files to specified format
"""

import json
import os
import glob
import re
from pathlib import Path

MAX_RECORDS = 20000

def process_jsonl_files():
    """Process all jsonl files and convert format"""

    # Source directory and target file paths
    trajectory_dir = "train/sft/data/afm/gemini_2.5_flash_random_2"
    target_file = "train/sft/LLaMA-Factory/data/deep_research/data_20.json"
    
    # Ensure target directory exists
    target_dir = os.path.dirname(target_file)
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all jsonl files
    jsonl_pattern = os.path.join(trajectory_dir, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    print(f"Found {len(jsonl_files)} .jsonl files")
    
    # Store all converted data
    converted_data = []
    total_records = 0
    error_count = 0
    generation_failed_count = 0
    
    # Process each jsonl file
    for file_path in jsonl_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        
                        # Check required fields
                        if 'input' not in data or 'response' not in data:
                            print(f"Warning: {os.path.basename(file_path)} line {line_num} missing required fields")
                            error_count += 1
                            continue

                        instruction_text = data["input"]
                        output_text = data["response"]
                        
                        # Convert to target format
                        converted_record = {
                            "instruction": instruction_text,
                            "input": "",  # Leave empty as required
                            "output": output_text
                        }
                        
                        converted_data.append(converted_record)
                        total_records += 1
                        if total_records >= MAX_RECORDS:
                            break
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {os.path.basename(file_path)} line {line_num}: {e}")
                        error_count += 1
                        continue

            if total_records >= MAX_RECORDS:
                break
                        
        except Exception as e:
            print(f"File reading error: {file_path}: {e}")
            error_count += 1
            continue
    
    print(f"Total processed {total_records} records")
    print(f"Encountered {error_count} errors")
    print(f"Generation failed {generation_failed_count} times")
    # Write to target file
    try:
        with open(target_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2, separators=(',', ': '))
        
        print(f"Data successfully saved to: {target_file}")
        print(f"File size: {os.path.getsize(target_file) / 1024 / 1024:.2f} MB")
        
        # Validate generated file
        print("Validating generated file...")
        with open(target_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            print(f"Validation successful: contains {len(test_data)} records")
            
            # Display brief information of first record
            if test_data:
                first_record = test_data[0]
                print(f"First record instruction length: {len(first_record['instruction'])}")
                print(f"First record output length: {len(first_record['output'])}")
        
    except Exception as e:
        print(f"File write error: {e}")
        return False
    
    return True

def main():
    print("Starting data reorganization...")
    success = process_jsonl_files()
    
    if success:
        print("Data reorganization completed!")
    else:
        print("Data reorganization failed!")

if __name__ == "__main__":
    main() 