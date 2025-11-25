import json
import re
from typing import List, Dict, Any, Union
from openai import OpenAI
import tiktoken


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(
                data, 
                file, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys
            )
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text))


def load_dataset(test_data_path):
    """
    Load dataset from local file.

    Args:
        test_data_path (str): Path to local test data JSON file
    """
    test_data = read_json(test_data_path)
    print(f"Loaded {len(test_data)} questions from dataset.")
    return test_data


def decode_response(response, default_judgement="incorrect"):
    """
    Decode JSON response from LLM.
    
    Args:
        response: The response string or dict from LLM
        default_judgement: Default judgement value if parsing fails.
                          Use "incorrect" for evaluation prompts, "no" for hack detection prompts.
    
    Returns:
        dict: Parsed JSON response or default dict with judgement
    """
    try:
        if isinstance(response, str):
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try direct JSON parsing
            return json.loads(response)
        return response
    except json.JSONDecodeError as e:
        # Try to extract JSON object from the response string
        if isinstance(response, str):
            json_match = re.search(r'\{[^{}]*"judgement"[^{}]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
        return {"judgement": default_judgement}
    except Exception as e:
        return {"judgement": default_judgement}
