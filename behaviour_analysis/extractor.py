#!/usr/bin/env python3
import json
import os
import argparse
import re
from pathlib import Path
from collections import defaultdict
import sys
from pathlib import Path
from typing import List
sys.path.append(str(Path(__file__).parent.parent))
from utils.token_calculator import tokenize
from prompt import *
import concurrent.futures
import threading
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig
import google.generativeai as generativeai
from tqdm import tqdm


MODEL_ID = "gemini-2.5-flash"
# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_question_id(filename):
    """Extract question ID from filename"""
    # Format: result_id.json, where id is a string that includes not only numbers but also letters
    # Convert Path object to string
    filename_str = str(filename) if hasattr(filename, '__str__') else filename
    match = re.search(r'(\w+)\.json', filename_str)
    return match.group(1) if match else None


def call_llm_with_id(question_id, prompt):
    """
    Call Gemini API to generate critique with question ID

    Args:
        question_id: Question ID for tracking
        prompt: Input prompt

    Returns:
        tuple: (question_id, thought, response)
    """
    max_try_times = 3
    for attempt in range(max_try_times):
        try:
            gemini_response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                )
            response = gemini_response.text
            return question_id, response

        except Exception as e:
            if attempt == max_try_times - 1:
                raise ValueError(f"Failed to generate response for question {question_id} after {max_try_times} tries: {e}")
            else:
                time.sleep(random.randint(1, 3))

    return question_id, response


def call_llm_batch_with_ids(id_prompt_list, max_workers=16):
    """
    Concurrently call LLM to generate multiple critiques with IDs

    Args:
        id_prompt_list: List of (question_id, prompt) tuples
        max_workers: Maximum number of concurrent workers

    Returns:
        dict: Dictionary mapping question_id to (thought, response)
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results
        futures = [executor.submit(call_llm_with_id, qid, prompt)
                  for qid, prompt in id_prompt_list]

        results = {}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing LLM calls"):
            try:
                question_id, response = future.result()
                results[question_id] = response
            except Exception as e:
                # Throw serious errors directly
                raise e

    return results


def get_extracted_behaviors(input_suffix: str, output_suffix: str) -> List[str] | None:

    # Ensure reasoner_input and reasoner_output directories exist
    extractor_input_path = Path(f"behaviour_analysis/reasoner_output/{input_suffix}")
    extractor_input_path.mkdir(parents=True, exist_ok=True)

    extractor_output_path = Path(f"behaviour_analysis/extractor_output/{output_suffix}")
    extractor_output_path.mkdir(parents=True, exist_ok=True)

    # Ensure consistent file order, use sorted() to maintain deterministic order
    reasoning_files = sorted(extractor_input_path.glob("*.json"))

    # Create a list containing IDs and prompts
    id_prompt_list = []
    for reasoning_file in reasoning_files:
        question_id = extract_question_id(reasoning_file)
        with open(reasoning_file, "r") as f:
            reasoning_data = json.load(f)
            reasoner_thinking = reasoning_data["reasoner_thinking"]
            reasoner_output = reasoning_data["reasoner_output"]
            reason_response = f"{reasoner_output}"
            prompt = extract_behavior_prompt_2.format(explanation_text=reason_response)

        id_prompt_list.append((question_id, prompt))

    # Use the new function to return a dictionary with IDs as keys
    results = call_llm_batch_with_ids(id_prompt_list)

    # Index results by ID to ensure no errors
    for question_id, response in results.items():
        try:
            # Try to validate the format with json load
            rules = json.loads(response.strip())

            # Save to the corresponding result_id.json in the extractor_output_path folder
            output_file = extractor_output_path / f"result_{question_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as e:
            print(f"JSON decode error for question {question_id}: {e}")



def main():
    """Main function to run the extractor"""
    parser = argparse.ArgumentParser(description="Extract rules from reasoning files")
    parser.add_argument("--input_suffix", type=str, default="",
                       help="Directory suffix for input paths (default: empty)")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Directory suffix for output paths (default: empty)")
    
    args = parser.parse_args()

    try:
        rules = get_extracted_behaviors(
            input_suffix=args.input_suffix,
            output_suffix=args.output_suffix,
        )

        print(f"Extraction completed. Results saved to behaviour_analysis/extractor_output/{args.output_suffix}/")

    except Exception as e:
        print(f"Error running extractor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    