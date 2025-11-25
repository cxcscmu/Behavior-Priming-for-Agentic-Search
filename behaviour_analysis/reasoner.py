#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
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
from google.genai.types import GenerateContentConfig, ThinkingConfig
import google.generativeai as generativeai
from prompt import reason_prompt

MODEL_ID = "gemini-2.5-flash"
# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)


def call_llm(prompt, question_id=None):
    """
    Call Gemini API to generate critique
    
    Args:
        prompt: Input prompt
        question_id: Question ID for logging
    
    Returns:
        str: LLM generated critique content, returns empty string on failure
    """    
    max_try_times = 3
    thought = ""
    response = ""
    for attempt in range(max_try_times):
        try:
            gemini_response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=GenerateContentConfig(
                        thinking_config=ThinkingConfig(include_thoughts=True),
                        max_output_tokens=10240
                    ),
                )
            for part in gemini_response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thought = part.text
                else:
                    response = part.text
            return thought, response
            
        except Exception as e:
            if attempt == max_try_times - 1:
                raise ValueError(f"Failed to generate response for question {question_id} after {max_try_times} tries: {e}")
            else:
                time.sleep(random.randint(1, 3))
    
    return thought, response


def call_llm_batch(reasoner_data_list, max_workers=16):
    """
    Concurrently call LLM to generate multiple critiques
    
    Args:
        reasoner_data_list: List of reasoner data, each element contains prompt and related information
        max_workers: Maximum number of concurrent workers
    
    Returns:
        list: List containing reasoner results
    """
    def process_single_reasoner(reasoner_data):
        """Process single reasoner"""
        question_id = reasoner_data['question_id']
        prompt = reasoner_data['reasoner_input']
        reasoner_thinking,reasoner_response = call_llm(prompt, question_id)
        print(f"generated reasoner_response for question {question_id}")
        reasoner_data['reasoner_thinking'] = reasoner_thinking
        reasoner_data['reasoner_output'] = reasoner_response
        return reasoner_data
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_reasoner, reasoner_data) 
                  for reasoner_data in reasoner_data_list]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing LLM calls"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Throw serious errors directly
                raise e
    
    return results

def reason(results_dir, logs_dir, questions, no_thinking=False, max_workers=32, use_llm=False):
    """
    Generate critique report, analyze multiple attempt trajectories and summarize lessons learned
    
    Args:
        results_dir: Results directory path
        logs_dir: Logs directory path  
        suffix: Model suffix name
        n: Number of attempts
        questions: List of questions
    """
    
    # Ensure reasoner_input and reasoner_output directories exist
    reasoner_input_path = Path(f"behaviour_analysis/reasoner_input")
    reasoner_input_path.mkdir(parents=True, exist_ok=True)
    
    reasoner_output_path = Path(f"behaviour_analysis/reasoner_output")
    reasoner_output_path.mkdir(parents=True, exist_ok=True)
    
    reasoner_data_list = []
    
    # Add progress bar for processing questions
    for question in tqdm(questions, desc="Processing questions"):
        question_text = question['question']
        ground_truth = question['answer']
        question_id = question['id']
        
        reasoner_input = ""
        
        correct_result_dir = Path(results_dir) / f"correct"
        incorrect_result_dir = Path(results_dir) / f"incorrect"
        correct_log_dir = Path(logs_dir) / f"correct"
        incorrect_log_dir = Path(logs_dir) / f"incorrect"
        
        correct_trajectory_file = correct_log_dir / f"trajectory_{question_id}.jsonl"
        correct_evaluation_file = correct_result_dir / f"result_{question_id}.json"
        incorrect_trajectory_file = incorrect_log_dir / f"trajectory_{question_id}.jsonl"
        incorrect_evaluation_file = incorrect_result_dir / f"result_{question_id}.json"
        
        # Check if files exist
        if not correct_trajectory_file.exists() or not correct_evaluation_file.exists():
            continue

        # if correct files exist, then incorrect files must exist
        assert incorrect_trajectory_file.exists() and incorrect_evaluation_file.exists(), f"Incorrect trajectory or evaluation file does not exist for question {question_id}"
        
        try:
            # Read trajectory file (JSONL format)
            correct_trajectory_content = ""
            with open(correct_trajectory_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for step_idx, line in enumerate(lines):
                    line_data = json.loads(line.strip())
                    agent_response = line_data.get('response', '')
                    if agent_response is not None and no_thinking and ("</think>" in agent_response):
                        agent_response = agent_response.split("</think>")[1]
                    environment_feedback = line_data.get('next_obs', '')
                    correct_trajectory_content += f"### Step {step_idx + 1}:\n\n#### Agent output: {agent_response}\n\n#### Environment Feedback: {environment_feedback}\n\n"

            # Read evaluation results
            with open(correct_evaluation_file, 'r', encoding='utf-8') as f:
                correct_evaluation_result = json.load(f)
                correct_score = correct_evaluation_result.get('score', 'N/A')
                if correct_score == 1:
                    correct_score = "Correct"
                elif correct_score == 0:
                    correct_score = "Incorrect"
                else:
                    correct_score = "N/A" 
            
            incorrect_trajectory_content = ""
            with open(incorrect_trajectory_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for step_idx, line in enumerate(lines):
                    line_data = json.loads(line.strip())
                    agent_response = line_data.get('response', '')
                    if agent_response is not None and no_thinking and ("</think>" in agent_response):
                        agent_response = agent_response.split("</think>")[1]
                    environment_feedback = line_data.get('next_obs', '')
                    incorrect_trajectory_content += f"### Step {step_idx + 1}:\n\n#### Agent output: {agent_response}\n\n#### Environment Feedback: {environment_feedback}\n\n"
            
            with open(incorrect_evaluation_file, 'r', encoding='utf-8') as f:
                incorrect_evaluation_result = json.load(f)
                incorrect_score = incorrect_evaluation_result.get('score', 'N/A')
                if incorrect_score == 1:
                    incorrect_score = "Correct"
                elif incorrect_score == 0:
                    incorrect_score = "Incorrect"
                else:
                    incorrect_score = "N/A"
                
            assert correct_score == "Correct" and incorrect_score == "Incorrect", f"Correct score {correct_score} or incorrect score {incorrect_score} is not correct for question {question_id}"
            
            # Format second part of prompt
            prompt = reason_prompt.format(question = question_text, trajectory_1 = correct_trajectory_content, trajectory_2 = incorrect_trajectory_content, evaluation_results_1 = correct_score, evaluation_results_2 = incorrect_score)
            reasoner_input = prompt
            
        except Exception as e:
            # Throw serious errors directly
            raise e
        
        if reasoner_input:            
            reasoner_input_file = reasoner_input_path / f"{question_id}.txt"
            with open(reasoner_input_file, 'w', encoding='utf-8') as f:
                f.write(reasoner_input)
            
            reasoner_input_length = tokenize(reasoner_input)
            
            reasoner_data = {
                "question_id": question_id,
                "question": question_text,
                "reasoner_input_length": reasoner_input_length,
                "reasoner_input": reasoner_input,
                "reasoner_thinking": "",
                "reasoner_output": ""  # Default empty, filled later by LLM
            }
            
            reasoner_data_list.append(reasoner_data)
    
    # Concurrently call LLM to generate critique
    if use_llm and reasoner_data_list:
        reasoner_data_list = call_llm_batch(reasoner_data_list, max_workers)
   

    # Save all individual question critique files
    for reasoner_data in tqdm(reasoner_data_list, desc="Saving results"):
        question_id = reasoner_data['question_id']
        reasoner_output_file = reasoner_output_path / f"{question_id}.json"
        with open(reasoner_output_file, 'w', encoding='utf-8') as f:
            json.dump(reasoner_data, f, ensure_ascii=False, indent=2)

    print(f"finish reason on {len(reasoner_data_list)} questions")
    
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate critique report to analyze agent trajectories")
    parser.add_argument("--results_dir", default="results/afm/behavior_analysis", help="Results directory path")
    parser.add_argument("--logs_dir", default="logs/afm/behavior_analysis", help="Logs directory path")
    parser.add_argument("--question_file", default="evaluation/short/afm/sft_full_qa.json", help="Question file path")
    parser.add_argument("--no_thinking", action="store_true", help="Whether to remove thinking process in trajectory")
    parser.add_argument("--use_llm", action="store_true", help="Whether to use real LLM to generate critique")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of threads for concurrent LLM calls")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    results_dir = args.results_dir
    logs_dir = args.logs_dir
    question_file = args.question_file
    # Check if question file exists
    if not os.path.exists(question_file):
        print(f"Error: Question file does not exist: {question_file}")
        exit(1)
    
    # Read question data
    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
    except Exception as e:
        raise Exception(f"Failed to read question file: {e}")
    
    # Run critique analysis
    reason(
        results_dir=results_dir, 
        logs_dir=logs_dir, 
        questions=questions,
        no_thinking=args.no_thinking,
        max_workers=args.max_workers,
        use_llm=args.use_llm,
    )