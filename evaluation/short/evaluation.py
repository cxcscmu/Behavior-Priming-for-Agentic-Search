import json
import re
import time
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from utils import read_json, write_json, load_dataset, decode_response
from tqdm import tqdm
from prompt import *

load_dotenv('keys.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=180)

def retry_predict(model, prompt, developer_prompt=None):
    messages = []
    if developer_prompt:
        messages.append({
            "role": "system",
            "content": developer_prompt
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            content = response.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            print(f"OpenAI API Error: {e}, retrying...")
    return ""


def evaluate_single_result_file(test_data, result_file_path, question_id, rerun=False, is_mhqa=False, enable_hack_detection=False, **kwargs):
    """
    Evaluate a single result file.

    Args:
        test_data (list): List of question data
        result_file_path (str): Path to result JSON file
        question_id (str): Question ID extracted from filename
        enable_hack_detection (bool): Whether to enable hack detection

    Returns:
        dict: Evaluation result or None if evaluation fails
    """
    try:
        result_data = read_json(result_file_path)
    except Exception as e:
        raise ValueError(f"Error reading file {result_file_path}: {e}")

    model_answer = result_data.get('answer', '')
    if not model_answer:
        raise ValueError(f"Warning: No prediction found in {result_file_path}")

    question_data = None
    for item in test_data:
        if str(item.get('id')) == str(question_id):
            question_data = item
            break

    if not question_data:
        raise ValueError(f"Warning: Could not find question data for ID {question_id}")

    question = question_data['question']
    golden_answer = question_data['answer']

    is_hack = False
    hack_rationale = ""
    if 'score' not in result_data or rerun:
        if enable_hack_detection:
            hack_evaluation_prompt = judge_hack_prompt.format(
                question=question,
                model_answer=model_answer
            )
            hack_output = retry_predict(
                "gpt-4o",
                hack_evaluation_prompt,
                developer_prompt="You are an evaluation assistant."
            )
            hack_json_output = decode_response(hack_output)
            if (hack_json_output and isinstance(hack_json_output, dict) and
                    "judgement" in hack_json_output and
                    hack_json_output['judgement'].lower() == "yes"):
                # detected hacking, skip the evaluation, set score to 0
                print(f"Hacking detected for {result_file_path}, set score to 0")
                is_hack = True
                score = 0
                hack_rationale = hack_json_output.get('rationale', '')
            else:
                # no hacking detected, continue the evaluation
                hack_rationale = hack_json_output.get('rationale', '') if hack_json_output and isinstance(hack_json_output, dict) else ''
                if is_mhqa: 
                    llm_evaluation_prompt = judge_prompt_mhqa.format(
                        question=question,
                        gt_answer=golden_answer,
                        pred_answer=model_answer
                    )
                else:
                    llm_evaluation_prompt = judge_prompt_web.format(
                        question=question,
                        gt_answer=golden_answer,
                        pred_answer=model_answer
                    )
                output = retry_predict(
                    "gpt-4o-mini",
                    llm_evaluation_prompt,
                    developer_prompt="You are an evaluation assistant."
                )
                json_output = decode_response(output)
                if (json_output and isinstance(json_output, dict) and
                    "judgement" in json_output and
                    json_output['judgement'].lower() == "correct"):
                    score = 1
                else:
                    score = 0
        else:
            # hack detection disabled, skip hack detection and proceed with normal evaluation
            hack_rationale = ""
            if is_mhqa: 
                llm_evaluation_prompt = judge_prompt_mhqa.format(
                    question=question,
                    gt_answer=golden_answer,
                    pred_answer=model_answer
                )
            else:
                llm_evaluation_prompt = judge_prompt_web.format(
                    question=question,
                    gt_answer=golden_answer,
                    pred_answer=model_answer
                )
            output = retry_predict(
                "gpt-4o-mini",
                llm_evaluation_prompt,
                developer_prompt="You are an evaluation assistant."
            )
            json_output = decode_response(output)
            if (json_output and isinstance(json_output, dict) and
                "judgement" in json_output and
                json_output['judgement'].lower() == "correct"):
                score = 1
            else:
                score = 0
        if 'score' in result_data:
            original_score = result_data['score']
            if original_score != score:
                print(f"Warning: Score mismatch for {result_file_path}, original score: {original_score}, new score: {score}")
    else:
        score = result_data['score']
        # Check if hack status was previously saved
        is_hack = result_data.get('is_hack', False)
        hack_rationale = result_data.get('hack_rationale', '')

    result_data['score'] = score
    result_data['ground_truth'] = golden_answer
    result_data['is_hack'] = is_hack
    result_data['hack_rationale'] = hack_rationale
    # Update the original file with evaluation result
    write_json(result_data, result_file_path)

    result = {
        'question_id': question_id,
        'question': question,
        'golden_answer': golden_answer,
        'prediction': model_answer,
        'score': score,
        'is_hack': is_hack,
        'file_path': result_file_path
    }

    return result


def evaluate_directory(test_data, results_dir_path, max_workers=8, rerun=False, is_mhqa=False, enable_hack_detection=False):
    """
    Evaluate all result files in a directory.

    Args:
        test_data (list): Loaded test data
        results_dir_path (str): Path to directory containing result files
        max_workers (int): Maximum number of concurrent workers
        enable_hack_detection (bool): Whether to enable hack detection
    """
    results_dir = Path(results_dir_path)
    all_result_files = list(results_dir.glob('result_*.json'))

    if not all_result_files:
        print(f"No result files found in {results_dir_path}")
        return None

    # Pre-filter result files based on existing question IDs in test_data
    valid_question_ids = {str(item.get('id')) for item in test_data if item.get('id') is not None}
    filtered_result_files = []
    skipped_files_count = 0

    for result_file in all_result_files:
        question_id = result_file.stem.replace('result_', '')
        if question_id in valid_question_ids:
            filtered_result_files.append(result_file)
        else:
            skipped_files_count += 1
    
    if not filtered_result_files:
        print(f"No valid result files to evaluate after filtering for test data. Skipped {skipped_files_count} files.")
        return None

    print(f"Found {len(filtered_result_files)} valid result files to evaluate (skipped {skipped_files_count} files not in test_data).")

    evaluated_results = []
    failed_evaluations = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(evaluate_single_result_file, test_data, str(result_file), result_file.stem.replace('result_', ''), rerun=rerun, is_mhqa=is_mhqa, enable_hack_detection=enable_hack_detection): result_file
            for result_file in filtered_result_files
        }
        for future in tqdm(as_completed(future_to_file), total=len(filtered_result_files), desc="Evaluating files"):
            result_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    evaluated_results.append(result)
                else:
                    failed_evaluations.append(str(result_file))
                    print(f"Failed to evaluate {result_file}")
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                failed_evaluations.append(str(result_file))

    if failed_evaluations:
        print(f"Failed to evaluate {len(failed_evaluations)} files")

    # Compute basic metrics
    if evaluated_results:
        total_count = len(evaluated_results)
        correct_count = sum(1 for result in evaluated_results if result.get('score') == 1)
        hack_count = sum(1 for result in evaluated_results if result.get('is_hack', False))
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        hack_rate = hack_count / total_count if total_count > 0 else 0.0

        metrics = {
            'total_count': total_count,
            'correct_count': correct_count,
            'accuracy_percentage': accuracy * 100,
            'hack_count': hack_count,
            'hack_rate_percentage': hack_rate * 100,
            'failed_count': len(failed_evaluations)
        }

        print(f"Evaluation completed: {metrics}")

        # Check if data_source field exists and compute per-source statistics
        data_source_stats = {}
        has_data_source = False
        
        # Create a mapping from question_id to data_source for quick lookup
        question_id_to_data_source = {}
        for item in test_data:
            if item.get('id') is not None and 'data_source' in item:
                question_id_to_data_source[str(item.get('id'))] = item['data_source']
                has_data_source = True
        
        if has_data_source:
            # Group results by data_source
            for result in evaluated_results:
                question_id = str(result.get('question_id'))
                data_source = question_id_to_data_source.get(question_id, 'unknown')
                
                if data_source not in data_source_stats:
                    data_source_stats[data_source] = {
                        'total_count': 0,
                        'correct_count': 0,
                        'accuracy_percentage': 0.0,
                        'hack_count': 0,
                        'hack_rate_percentage': 0.0
                    }
                
                data_source_stats[data_source]['total_count'] += 1
                if result.get('score') == 1:
                    data_source_stats[data_source]['correct_count'] += 1
                if result.get('is_hack', False):
                    data_source_stats[data_source]['hack_count'] += 1
            
            # Calculate accuracy and hack rate for each data_source
            for data_source in data_source_stats:
                stats = data_source_stats[data_source]
                stats['accuracy_percentage'] = (stats['correct_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0.0
                stats['hack_rate_percentage'] = (stats['hack_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0.0
            
            # Print detailed statistics by data_source
            print("\n=== Statistics by Data Source ===")
            for data_source in sorted(data_source_stats.keys()):
                stats = data_source_stats[data_source]
                print(f"{data_source}: {stats['correct_count']}/{stats['total_count']} ({stats['accuracy_percentage']:.2f}%), Hack: {stats['hack_count']}/{stats['total_count']} ({stats['hack_rate_percentage']:.2f}%)")
            
            # Add data_source statistics to metrics
            metrics['data_source_statistics'] = data_source_stats

    return 


def main():
    parser = argparse.ArgumentParser(description='Evaluate results from directory.')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--data_path', type=str, default='evaluation/short/test.json', help='Test data path')
    parser.add_argument('--rerun', action='store_true', help='Rerun evaluation')
    parser.add_argument('--mhqa', action='store_true', help='Evaluate on MHQA dataset')
    parser.add_argument('--enable_hack_detection', action='store_true', help='Enable hack detection')
    args = parser.parse_args()

    results_dir = args.results_dir
    test_data_path = args.data_path
    is_mhqa = args.mhqa
    # Load test data
    test_data = load_dataset(test_data_path)

    # Evaluate results from directory
    evaluate_directory(
        test_data,
        results_dir,
        max_workers=64,
        rerun=args.rerun,
        is_mhqa=is_mhqa,
        enable_hack_detection=args.enable_hack_detection
    )

if __name__ == "__main__":
    main()