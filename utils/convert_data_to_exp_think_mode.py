#!/usr/bin/env python3
"""
Script to convert prompts in JSONL files from internal thinking mode to explicit thinking mode
"""

import json
import os
import glob
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prompt import short_answer_prompt_internal_thinking_base, short_answer_prompt_explicit_thinking_base

def convert_prompt_content(input_text):
    """
    Convert internal thinking prompt to explicit thinking prompt
    Only replace the part before Question:
    """
    # Prompt content imported from prompt.py, remove {question} placeholder and Question: part
    internal_thinking_part = short_answer_prompt_internal_thinking_base.replace("{question}", "").replace("Question: \n", "")
    explicit_thinking_part = short_answer_prompt_explicit_thinking_base.replace("{question}", "").replace("Question: \n", "")

    # Find the position of Question:
    question_pos = input_text.find("Question:")
    if question_pos == -1:
        print("Warning: Question: marker not found, skipping this file")
        return input_text
    
    # Extract the part before Question:
    before_question = input_text[:question_pos]
    
    # Check if it matches internal thinking prompt
    if before_question.strip() == internal_thinking_part.strip():
        # Replace with explicit thinking prompt
        after_question = input_text[question_pos:]
        return explicit_thinking_part + after_question
    else:
        print("Warning: internal thinking prompt not found, skipping this file")
        return input_text

def process_jsonl_file(input_file, output_file):
    """
    Process a single JSONL file
    """
    print(f"Processing file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            if 'input' in data:
                original_input = data['input']
                converted_input = convert_prompt_content(original_input)
                data['input'] = converted_input
                converted_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                print(f"Warning: Line {line_num} has no 'input' field, skipping")
                converted_lines.append(line)
        except json.JSONDecodeError as e:
            print(f"Error: JSON parsing failed at line {line_num}: {e}")
            converted_lines.append(line)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(converted_lines)
    
    print(f"Completed: {output_file}")

def main():
    """
    Main function
    """
    # Input directory
    input_dir = "train/sft/data/afm_mhqa/gemini_2.5_flash_random"
    # Output directory
    output_dir = "train/sft/data/afm_mhqa/gemini_2.5_flash_random_explicit_thinking"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .jsonl files
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in directory {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Process each file
    for input_file in jsonl_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        process_jsonl_file(input_file, output_file)
    
    print(f"Conversion completed! Output directory: {output_dir}")

if __name__ == "__main__":
    main()
