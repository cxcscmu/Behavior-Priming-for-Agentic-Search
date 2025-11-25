#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from prompt import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.token_calculator import tokenize
import concurrent.futures
import threading
import time
import random
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types
import google.generativeai as generativeai


MODEL_ID = "gemini-2.5-pro"
# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)


def call_llm(prompt):
    """
    Call Gemini API to generate critique
    
    Args:
        prompt: Input prompt
    
    Returns:
        str: LLM generated critique content, returns empty string on failure
    """    
    max_try_times = 3
    for attempt in range(max_try_times):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            if "context" in str(e).lower() or "length" in str(e).lower():
                raise ValueError(f"Context length error: {e}")
            
            if attempt == max_try_times - 1:
                return ""  # Return empty string on failure
            else:
                time.sleep(random.randint(1, 3))
    
    return ""

def merge_single_behavior_list(behavior_list):
    list_str = ""
    for i, behavior in enumerate(behavior_list):
        list_str += f"{i+1}. {behavior}\n"
    prompt = merge_prompt_1.format(behaviors_text=list_str)
    return call_llm(prompt)


def merge_chunk_concurrent(chunk_data):
    """Process a single chunk concurrently"""
    chunk_idx, chunk = chunk_data
    max_try_times = 3
    for attempt in range(max_try_times):
        try:
            merged_text = merge_single_behavior_list(chunk)
            merged_behaviors = parse_json_response(merged_text)
            if merged_behaviors:
                return chunk_idx, merged_behaviors, None
        except Exception as e:
            if attempt == max_try_times - 1:
                return chunk_idx, chunk, f"Error merging chunk {chunk_idx+1}: {e}"
            else:
                time.sleep(random.randint(1, 3))
    
    return chunk_idx, chunk, f"Failed to merge chunk {chunk_idx+1} after {max_try_times} attempts"


def load_all_behaviors(input_dir):
    """Load all behaviors from JSON files in the input directory"""
    all_behaviors = []
    for json_file in Path(input_dir).glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                behaviors = json.load(f)
                if isinstance(behaviors, list):
                    all_behaviors.extend(behaviors)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    return all_behaviors


def parse_json_response(response_text):
    """Parse JSON response from LLM, return list or empty list on failure"""
    if not response_text.strip():
        return []

    try:
        # Try to parse as JSON array directly
        result = json.loads(response_text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text (in case of extra content)
    import re
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"Failed to parse JSON response: {response_text[:200]}...")
    return []


def merge_behaviors_iteratively(behaviors, max_behavior=100, min_behavior=32):
    """Iteratively merge behaviors until the number is less than min_behavior"""
    current_behaviors = behaviors.copy()
    iteration = 1

    while len(current_behaviors) > min_behavior:
        print(f"Iteration {iteration}: Processing {len(current_behaviors)} behaviors")

        # Split behaviors into chunks of max_behavior size
        chunks = []
        for i in range(0, len(current_behaviors), max_behavior):
            chunks.append(current_behaviors[i:i + max_behavior])

        print(f"Split into {len(chunks)} chunks")

        # Merge each chunk concurrently
        merged_results = []
        chunk_data_list = [(i, chunk) for i, chunk in enumerate(chunks)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(merge_chunk_concurrent, chunk_data) for chunk_data in chunk_data_list]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Merging chunks (iteration {iteration})"):
                try:
                    chunk_idx, result, error = future.result()
                    if error:
                        print(f"Warning: {error}")
                        merged_results.extend(result)  # Keep original behaviors on failure
                    else:
                        merged_results.extend(result)
                except Exception as e:
                    print(f"Unexpected error processing chunk: {e}")
                    # Fallback to original chunk
                    chunk_idx = futures.index(future)
                    merged_results.extend(chunks[chunk_idx])

        current_behaviors = merged_results
        print(f"After iteration {iteration}: {len(current_behaviors)} behaviors")
        iteration += 1

        # Safety check to prevent infinite loops
        if iteration > 10:
            print("Warning: Too many iterations, stopping")
            break

    return current_behaviors


def main():
    """Main function to load behaviors and perform iterative merging"""
    print("Loading all behaviors from input directory...")
    all_behaviors = load_all_behaviors(input_dir)
    print(f"Loaded {len(all_behaviors)} behaviors from {len(list(Path(input_dir).glob('*.json')))} files")

    if not all_behaviors:
        print("No behaviors found!")
        return

    print("Starting iterative merging...")
    final_behaviors = merge_behaviors_iteratively(all_behaviors, max_behavior=MAX_BEHAVIOR, min_behavior=MIN_BEHAVIOR)

    print(f"Final result: {len(final_behaviors)} behaviors")

    # Save final result
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "merged_behaviors.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_behaviors, f, indent=2, ensure_ascii=False)

    print(f"Saved final merged behaviors to: {output_file}")

    return final_behaviors

input_dir = "behaviour_analysis/extractor_output/2"
output_dir = "behaviour_analysis/merger_output/7"

MAX_BEHAVIOR = 250
MIN_BEHAVIOR = 5

if __name__ == "__main__":
    main()

