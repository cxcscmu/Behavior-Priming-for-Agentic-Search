#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

def evaluate_best_of_n(n, suffix, base_dir="results/webwalkerqa/best_of_n", questions_file="evaluation/short/webwalkerqa/test_random_250.json"):
    """
    evaluate best-of-n results, calculate pass@n metrics
    
    Args:
        n (int): number of trials to consider for pass@n
        suffix (str): prefix of trial directory names
        base_dir (str): best_of_n results directory path
        questions_file (str): path to questions file
        
    Returns:
        dict: pass@n metrics
    """
    base_path = Path(base_dir)
    
    # load questions data
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # create mapping of question id to question info
    questions_dict = {q['id']: q for q in questions_data}
    total_questions = len(questions_dict)
    print(f"loaded {total_questions} questions")
    
    # collect trial directories in order (suffix_1, suffix_2, ..., suffix_n)
    trial_dirs = []
    max_trial_num = 0
    
    # first, find the maximum trial number available
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith(suffix + "_"):
            try:
                trial_num = int(d.name[len(suffix) + 1:])
                max_trial_num = max(max_trial_num, trial_num)
            except ValueError:
                continue
    
    if max_trial_num == 0:
        print(f"no trial directories found in {base_dir}")
        return {}
    
    # only use the first n trial directories in order
    selected_trial_dirs = []
    for i in range(1, min(n + 1, max_trial_num + 1)):
        trial_dir = base_path / f"{suffix}_{i}"
        if trial_dir.exists() and trial_dir.is_dir():
            selected_trial_dirs.append(trial_dir)
        else:
            print(f"warning: trial directory {trial_dir.name} does not exist")
    
    if not selected_trial_dirs:
        print(f"no valid trial directories found for pass@{n}")
        return {}
    
    actual_n = len(selected_trial_dirs)
    
    # collect scores for each question across selected trials in order
    question_scores = defaultdict(list)
    
    for trial_dir in selected_trial_dirs:
        # iterate through all questions and try to find their result files
        for question_id in questions_dict.keys():
            result_file = trial_dir / f"result_{question_id}.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    score = result_data.get('score', 0)
                    question_scores[question_id].append(score)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"warning: error reading {result_file}: {e}")
                    question_scores[question_id].append(0)
            else:
                # if result file doesn't exist, assume score is 0
                question_scores[question_id].append(0)
    
    # calculate pass@n metrics (best results for each question across the selected trials)
    best_scores = []
    missing_questions = []
    
    for question_id in sorted(questions_dict.keys()):
        scores = question_scores.get(question_id, [])
        if len(scores) >= actual_n:
            # get highest score for this question across all selected trials
            best_score = max(scores) if scores else 0
            best_scores.append(best_score)
        else:
            # not enough trials for this question (should not happen with our logic)
            missing_questions.append(question_id)
            if scores:
                # use available scores
                best_score = max(scores)
                best_scores.append(best_score)
            else:
                # no results for this question
                best_scores.append(0)
    
    if missing_questions:
        print(f"warning: {len(missing_questions)} questions have fewer than {actual_n} trials")
    
    if not best_scores:
        print("no valid evaluation results found")
        return {}
    
    # calculate overall metrics
    total_questions = len(best_scores)
    correct_count = sum(1 for score in best_scores if score == 1.0)
    pass_at_n = correct_count / total_questions if total_questions > 0 else 0.0
    
    results = {
        'total_questions': total_questions,
        'correct_count': correct_count,
        'n': n,
        'actual_n': actual_n,
        'pass_at_n': pass_at_n,
        'pass_at_n_percentage': pass_at_n * 100,
        'total_num_trials': actual_n,
        'missing_questions_count': len(missing_questions)
    }
    
    print(f"\n=== Best-of-N (Pass@{n}) results ===")
    print(f"correct answers: {correct_count}")
    print(f"Pass@{n}: {pass_at_n:.4f} ({pass_at_n * 100:.2f}%)")

    
    return results


if __name__ == "__main__":
    # run evaluation
    suffix = "behavior_200"

    for n in [1, 2, 4, 8]:
        results = evaluate_best_of_n(n=n, suffix=suffix)
       
