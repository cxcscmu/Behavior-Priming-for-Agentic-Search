import os
from pathlib import Path
import time
import json
import statistics

def analyze_json_files(directory_path):
    """Analyze JSON files in the specified directory and calculate statistics"""
    path = Path(directory_path)
    
    if not path.exists():
        print(f"Error: Directory '{directory_path}' does not exist")
        return {}
    
    if not path.is_dir():
        print(f"Error: '{directory_path}' is not a directory")
        return {}
    
    # Find all JSON files
    json_files = list(path.glob("*.json"))
    json_count = len(json_files)
    
    if json_count == 0:
        print(f"No JSON files found in directory: {directory_path}")
        return {}
    
    # Initialize data collectors
    all_data = {
        'turns': [],
        'search_count': [],
        'summary_count': [],
        'score': [],
        'context_lengths_avg': []  # Average context length per file
    }
    
    valid_files = 0

    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract numeric fields
            if 'turns' in data and isinstance(data['turns'], (int, float)):
                all_data['turns'].append(data['turns'])
            
            if 'search count' in data and isinstance(data['search count'], (int, float)):
                all_data['search_count'].append(data['search count'])
            
            if 'summary count' in data and isinstance(data['summary count'], (int, float)):
                all_data['summary_count'].append(data['summary count'])

            if 'score' in data and isinstance(data['score'], (int, float)):
                all_data['score'].append(data['score']*100)
            
            # Calculate average context length for this file
            if 'context lengths' in data and isinstance(data['context lengths'], list):
                context_lengths = [x for x in data['context lengths'] if isinstance(x, (int, float))]
                if context_lengths:
                    avg_context = statistics.mean(context_lengths)
                    all_data['context_lengths_avg'].append(avg_context)
            
            valid_files += 1
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process file {json_file.name}: {e}")
            continue
    
    # Calculate and display statistics
    print(f"Directory: {directory_path}")
    print(f"Total JSON files: {json_count}")
    print(f"Valid files processed: {valid_files}")
    print("-" * 50)
    
    results = {}
    
    for field_name, values in all_data.items():
        if values:
            avg_value = statistics.mean(values)
            results[field_name] = {
                'average': avg_value,
                'count': len(values),
                'min': min(values),
                'max': max(values)
            }
            
            field_display = field_name.replace('_', ' ').title()
            print(f"{field_display}:")
            print(f"  Average: {avg_value:.2f}")
            print(f"  Range: {min(values)} - {max(values)}")
            print()
    
    return results

def count_json_files(directory_path):
    """Legacy function for backward compatibility"""
    results = analyze_json_files(directory_path)
    return len(list(Path(directory_path).glob("*.json"))) if Path(directory_path).exists() else 0

if __name__ == "__main__":
    # Specify the directory to analyze
    target_directory = "results/webwalkerqa/best_of_n/random_200_1"
    
    # Run the enhanced analysis
    analyze_json_files(target_directory)