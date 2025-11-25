#!/usr/bin/env python3
"""
Script to calculate scores by level for GAIA evaluation results.
Each question in the test dataset has a Level field (1, 2, or 3).
This script aggregates scores by level for a specific model folder.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import re


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_question_id_from_filename(filename):
    """Extract question ID from result filename (e.g., 'result_1.json' -> '1')."""
    match = re.match(r'result_(\d+)\.json', filename)
    return match.group(1) if match else None


def get_question_level(test_data, question_id):
    """Get the level for a specific question ID."""
    for item in test_data:
        if str(item.get('id')) == str(question_id):
            return item.get('Level')
    return None


def calculate_scores_by_level(test_data_path, results_dir_path):
    """
    Calculate scores by level for a specific model folder.
    
    Args:
        test_data_path (str): Path to test.json file containing questions and levels
        results_dir_path (str): Path to directory containing result_*.json files
    
    Returns:
        dict: Results organized by level
    """
    # Load test data
    test_data = load_json(test_data_path)
    
    # Initialize counters by level
    level_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'scores': []})
    
    # Get all result files
    results_dir = Path(results_dir_path)
    result_files = list(results_dir.glob('result_*.json'))
    
    if not result_files:
        print(f"No result files found in {results_dir_path}")
        return None
    
    print(f"Processing {len(result_files)} result files from {results_dir_path}")
    
    # Process each result file
    for result_file in result_files:
        try:
            # Extract question ID from filename
            question_id = extract_question_id_from_filename(result_file.name)
            if not question_id:
                print(f"Warning: Could not extract question ID from {result_file.name}")
                continue
            
            # Load result data
            result_data = load_json(result_file)
            
            # Get score from result file
            score = result_data.get('score')
            if score is None:
                print(f"Warning: No score found in {result_file.name}")
                continue
            
            # Get level for this question
            level = get_question_level(test_data, question_id)
            if level is None:
                print(f"Warning: Could not find level for question ID {question_id}")
                continue
            
            # Update statistics
            level_stats[level]['total'] += 1
            level_stats[level]['scores'].append(score)
            if score == 1:
                level_stats[level]['correct'] += 1
                
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    # Calculate accuracy for each level
    results = {}
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0.0
        
        results[f'Level_{level}'] = {
            'total_questions': stats['total'],
            'correct_answers': stats['correct'],
            'accuracy_percentage': round(accuracy, 2),
            'scores': stats['scores']
        }
    
    # Calculate overall statistics
    total_questions = sum(stats['total'] for stats in level_stats.values())
    total_correct = sum(stats['correct'] for stats in level_stats.values())
    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0.0
    
    results['Overall'] = {
        'total_questions': total_questions,
        'correct_answers': total_correct,
        'accuracy_percentage': round(overall_accuracy, 2)
    }
    
    return results


def print_results(results, model_name):
    """Print results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"GAIA Evaluation Results for: {model_name}")
    print(f"{'='*60}")
    
    # Print level-wise results
    for level_key in sorted([k for k in results.keys() if k.startswith('Level_')]):
        level_num = level_key.split('_')[1]
        stats = results[level_key]
        
        print(f"\nLevel {level_num}:")
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"  Correct Answers: {stats['correct_answers']}")
        print(f"  Accuracy: {stats['accuracy_percentage']}%")
    
    # Print overall results
    overall = results['Overall']
    print(f"\nOverall Performance:")
    print(f"  Total Questions: {overall['total_questions']}")
    print(f"  Correct Answers: {overall['correct_answers']}")
    print(f"  Accuracy: {overall['accuracy_percentage']}%")
    print(f"{'='*60}")


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate GAIA evaluation scores by level for a specific model folder.'
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results/gaia/random_sft',
        help='Path to directory containing result_*.json files for a specific model'
    )
    parser.add_argument(
        '--test_data', 
        type=str, 
        default='evaluation/short/gaia/test.json',
        help='Path to test.json file containing questions and levels'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='Path to save results JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.test_data):
        print(f"Error: Test data file not found: {args.test_data}")
        return
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Calculate scores by level
    results = calculate_scores_by_level(args.test_data, args.results_dir)
    
    if results is None:
        print("No results to display.")
        return
    
    # Extract model name from results directory path
    model_name = os.path.basename(args.results_dir.rstrip('/'))
    
    # Print results
    print_results(results, model_name)
    
    # Save results if output path provided
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()