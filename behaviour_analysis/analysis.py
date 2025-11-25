#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
# Make project root available on sys.path (for symmetric behavior with reasoner.py)
import sys
import random
from pathlib import Path as _Path
import traceback
sys.path.append(str(_Path(__file__).parent.parent))
from prompt import *  
from openai import OpenAI
from dotenv import load_dotenv  # type: ignore
from google import genai  # type: ignore
from google.genai.types import GenerateContentConfig, ThinkingConfig, HttpOptions  # type: ignore
from pydantic import BaseModel  # type: ignore


MODEL_ID = "gemini-2.5-flash"

def parse_log_file(log_path: str, add_thinking: bool = True) -> Tuple[str, List[str]]:
    """
    Parse a trajectory jsonl file to extract trajectory text.
    """
    trajectory_content = ""
    step_content = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for step_idx, line in enumerate(lines):
            line_data = json.loads(line.strip())
            agent_response = line_data.get('response', '')
            if agent_response is not None and (not add_thinking) and ("</think>" in agent_response):
                agent_response = agent_response.split("</think>")[1]
            environment_feedback = line_data.get('next_obs', '')
            step_content_str = f"### Step {step_idx + 1}:\n\n#### Agent output: {agent_response}\n\n#### Environment Feedback: {environment_feedback}"
            trajectory_content += f"{step_content_str}\n\n"
            step_content.append(step_content_str)

    return trajectory_content, step_content

def load_result_score(result_path: str) -> int:
    """Return 1 for correct, 0 for incorrect, else raise."""
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if 'score' not in data:
            raise ValueError(f"score not in {result_path}")
        
        score = data.get("score")
        
        if score not in (0, 1):
            raise ValueError(f"Invalid score in {result_path}: {score}")
    except Exception as e:
        print(f"Error loading score from {result_path}: {e}")
        return 0
    return int(score)

def build_prompt(question: str, content: str) -> str:

    prompt = judge_behavior_prompt.format(question=question, trajectory=content)
    
    return prompt

def init_llm_client() -> Tuple[OpenAI, str]:
    
    load_dotenv("keys.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment or keys.env")
    client = genai.Client(api_key=api_key)
    return (client, MODEL_ID)

class BehaviorJudge(BaseModel):
    behavior1: str
    behavior2: str
    behavior3: str
    behavior4: str

def call_llm(client: OpenAI, model_name: str, prompt: str, max_try_times: int = 3) -> Tuple[str, str]:
    last_err = None
    last_traceback = None
    for _ in range(max_try_times):
        try:
            thought = ""
            response = ""
            response_schema = BehaviorJudge
            
            # Truncate if too long
            if len(prompt) > 60000:
                prompt = prompt[:60000]

            gemini_response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    thinking_config=ThinkingConfig(include_thoughts=True),
                    max_output_tokens=5120,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            # Extract thought text if present
            try:
                for part in gemini_response.candidates[0].content.parts:
                    if not getattr(part, "text", None):
                        continue
                    if getattr(part, "thought", False):
                        thought = part.text
            except Exception as ex:
                # Avoid referencing undefined variable in logs
                print(f"failed to extract thought from gemini, error: {ex}")

            # Prefer structured parsed result
            parsed_obj = getattr(gemini_response, "parsed", None)
            if parsed_obj is not None:
                if hasattr(parsed_obj, "model_dump"):
                    parsed_dict = parsed_obj.model_dump()
                elif hasattr(parsed_obj, "dict"):
                    parsed_dict = parsed_obj.dict()
                else:
                    parsed_dict = parsed_obj  # Fallback
                response = json.dumps(parsed_dict, ensure_ascii=False)
            else:
                # Fallback to text if parsed unavailable
                response = getattr(gemini_response, "text", "")

            return thought, response
        except Exception as e:
            last_err = e
            last_traceback = traceback.format_exc()
    raise RuntimeError(f"LLM call failed after {max_try_times} attempts: {last_err}\n{last_traceback}")


def _extract_question_id(filename):
    match = re.search(r'result_(\w+)\.json', filename)
    return match.group(1) if match else None

def _extract_first_json_block(text: str) -> str:
    import re
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    s = t.find("{")
    if s != -1:
        depth = 0
        for i, ch in enumerate(t[s:], start=s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[s:i+1]
    return t

def analyze_trajectory(
    client: OpenAI,
    model_name: str,
    question_id: str,
    question: str,
    ground_truth: str,
    trajectory: str,
) -> Optional[Dict[str, str]]:
    # Single prompt returns a merged JSON for all behaviors
    prompt = build_prompt(question, trajectory)
    analysis_result = None
    
    for attempt in range(5):
        try:
            thought, llm_out = call_llm(client=client, model_name=model_name, prompt=prompt, max_try_times=3)
            block = _extract_first_json_block(llm_out)
            if not block or not block.strip().startswith("{"):
                raise ValueError("empty or non-JSON output from LLM")
            analysis_result = json.loads(block)
            break
        except Exception as e:
            if attempt == 4:
                preview = (llm_out or "")[:200].replace("\n", " ") if 'llm_out' in locals() else ""
                print(f"failed to judge behavior, question_id: {question_id}, error: {e}, preview: {preview}")
    
    return analysis_result

def list_result_ids(results_dir: str) -> List[str]:
    ids: List[str] = []
    for entry in os.listdir(results_dir):
        id = _extract_question_id(entry)
        ids.append(id)
    return sorted(ids)

def load_questions(question_path: str) -> Dict[str, str]:
    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    question_dict = {}
    for question in questions:
        question_dict[question["id"]] = {
            "question": question["question"],
            "ground_truth": question["answer"]
        }
    return question_dict


def prepare_questions_and_client(logs_dir: str, results_dir: str, question_path: str):
    questions = load_questions(question_path)
    client, model_name = init_llm_client()

    kept_questions = {}
    for question_id in questions.keys():
        log_path = os.path.join(logs_dir, f"trajectory_{question_id}.jsonl")
        result_path = os.path.join(results_dir, f"result_{question_id}.json")
        if os.path.exists(log_path) and os.path.exists(result_path):
            kept_questions[question_id] = questions[question_id]
    
    print(f"Get {len(kept_questions)} questions with logs and results")
    return kept_questions, client, model_name

def process_trajectories_parallel(logs_dir: str, results_dir: str, questions: Dict[str, str], 
                                 client: OpenAI, model_name: str, max_workers: int, rerun: bool):
    def process_single(logs_dir, results_dir, question_id: str, rerun: bool) -> Tuple[str, int, Optional[Dict[str, str]]]:
        log_path = os.path.join(logs_dir, f"trajectory_{question_id}.jsonl")
        result_path = os.path.join(results_dir, f"result_{question_id}.json")
        
        trajectory, steps = parse_log_file(log_path)
        score = load_result_score(result_path)
        question = questions[question_id]["question"]
        ground_truth = questions[question_id]["ground_truth"]

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading result file {result_path}: {e}")
            return question_id, score, None

        if "analysis_result" in data and data["analysis_result"] is not None and not rerun:
            analysis_result = data.get("analysis_result")
        else:
            analysis_result = analyze_trajectory(
                client=client,
                model_name=model_name,
                question_id=question_id,
                question=question,
                ground_truth=ground_truth,
                trajectory=trajectory,
            )
            
        # Save per-trajectory judgments (judgments is now a dict in all_together mode)
        data["analysis_result"] = analysis_result
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return question_id, score, analysis_result

    # Parallel over trajectories
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single, logs_dir, results_dir, question_id, rerun) for question_id in questions.keys()]
        with tqdm(total=len(futures), desc="Analyzing trajectories") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    question_id, score, analysis_result = fut.result()
                except Exception as e:
                    pbar.update(1)
                    raise
                else:
                    pbar.update(1)

def analyze_and_report_results(results_dir: str, questions: Dict[str, str]):
    # Load all the detailed results
    total_dict: Dict[str, int] = {} # behavior -> number
    correct_dict: Dict[str, int] = {} # behavior -> number
    incorrect_dict: Dict[str, int] = {} # behavior -> number
    
    total_all_behavior_number = 0
    correct_all_behavior_number = 0
    incorrect_all_behavior_number = 0
    
    total_trajectory_number = 0
    correct_trajectory_number = 0
    incorrect_trajectory_number = 0
    valid_trajectory_number = 0  # analysis present
    analysis_success_number = 0  # analysis success
    initialized = False
    
    for question_id in questions.keys():
        result_path = os.path.join(results_dir, f"result_{question_id}.json")
        if not os.path.exists(result_path):
            continue
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading result file {result_path}: {e}")
            continue
        
        total_trajectory_number += 1
        score = data.get("score")

        if score == 1:
            correct_trajectory_number += 1
        else:
            incorrect_trajectory_number += 1

        analysis_result = data.get("analysis_result")

        ok = isinstance(analysis_result, dict) and len(analysis_result) > 0

        if ok:
            analysis_success_number += 1

        if not ok:
            # Skip aggregation for this trajectory when any side failed
            continue

        if not initialized:
            behavior_keys = list(analysis_result.keys())
            for behavior_key in behavior_keys:
                total_dict[behavior_key] = 0
                correct_dict[behavior_key] = 0
                incorrect_dict[behavior_key] = 0
            initialized = True

        valid_trajectory_number += 1
        all_behavior = True

        for behavior_key in behavior_keys:
            behavior_result = analysis_result.get(behavior_key, "No")
            if str(behavior_result).lower() == "yes":
                total_dict[behavior_key] += 1
                if score == 1:
                    correct_dict[behavior_key] += 1
                else:
                    incorrect_dict[behavior_key] += 1
            else:
                all_behavior = False

        if all_behavior:
            total_all_behavior_number += 1
            if score == 1:
                correct_all_behavior_number += 1
            else:
                incorrect_all_behavior_number += 1

    # Print per-behavior ratios only if we had valid judgments
    if initialized and valid_trajectory_number > 0:
        for behavior_key in behavior_keys:
            print(f"{behavior_key} yes ratio: {total_dict[behavior_key] / valid_trajectory_number * 100:.2f}%")
        print(f"all behavior ratio: {total_all_behavior_number / valid_trajectory_number * 100:.2f}%")

        if correct_trajectory_number > 0:
            print("\n--- Correct Trajectories ---")
            for behavior_key in behavior_keys:
                print(f"Correct - {behavior_key} yes ratio: {correct_dict[behavior_key] / correct_trajectory_number * 100:.2f}%")
            print(f"Correct - all behavior ratio: {correct_all_behavior_number / correct_trajectory_number * 100:.2f}%")

        if incorrect_trajectory_number > 0:
            print("\n--- Incorrect Trajectories ---")
            for behavior_key in behavior_keys:
                print(f"Incorrect - {behavior_key} yes ratio: {incorrect_dict[behavior_key] / incorrect_trajectory_number * 100:.2f}%")
            print(f"Incorrect - all behavior ratio: {incorrect_all_behavior_number / incorrect_trajectory_number * 100:.2f}%")
    else:
        print("No valid judgments to aggregate behavior ratios.")

    # Print judge success rates over all trajectories
    if total_trajectory_number > 0:
        analysis_rate = analysis_success_number / total_trajectory_number * 100
        valid_rate = (valid_trajectory_number / total_trajectory_number) * 100
        print(f"valid trajectories: {valid_trajectory_number}/{total_trajectory_number} ({valid_rate:.2f}%)")
        print(f"behavior analysis successful: {analysis_success_number}/{total_trajectory_number} ({analysis_rate:.2f}%)")
        
    else:
        print("No trajectories found to compute success rates.")

def main():
    parser = argparse.ArgumentParser(description="Analyze behavior on trajectories using LLM judge")
    parser.add_argument(
        "--logs_dir",
        default="logs/afm/behavior_analysis/incorrect",
        help="Dataset name",
    )
    parser.add_argument(
        "--results_dir",
        default="results/afm/behavior_analysis/incorrect",
        help="Directory containing concise trajectory markdown logs",
    )
    parser.add_argument(
        "--question_path",
        default="evaluation/short/afm/sft_full_qa.json",
        help="Dataset name",
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun evaluation")
    parser.add_argument("--max_workers", type=int, default=60, help="Thread pool size for parallel evaluation")

    args = parser.parse_args()

    logs_dir = args.logs_dir
    results_dir = args.results_dir
    question_path = args.question_path
    rerun = args.rerun
    max_workers = args.max_workers

    if not os.path.exists(logs_dir) or not os.path.exists(results_dir) or not os.path.exists(question_path):    
        print(f"logs_dir or results_dir or question_path does not exist: {logs_dir} or {results_dir} or {question_path}")
        exit(1)

    print(f"logs_dir: {logs_dir}")
    print(f"results_dir: {results_dir}")
    print(f"question_path: {question_path}")
    
    questions, client, model_name = prepare_questions_and_client(logs_dir, results_dir, question_path)
    
    process_trajectories_parallel(logs_dir, results_dir, questions, client, model_name, max_workers, rerun)
    
    analyze_and_report_results(results_dir, questions)


if __name__ == "__main__":
    main()
