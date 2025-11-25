'''
Arguments:
    --batch_file: path to json file containing array of id and question fields
    --long_report: generate long report (default is short answer)
    --answer_dir: result directory
    --log_dir: log directory

Logs:
- see all system messages in /logs
    - /logs/search_{question_id}.log: search query and results
    - /logs/trajectory_{question_id}.md: total trajectory of the agent

'''

import re, os, requests
import boto3
import argparse
from datetime import datetime
from collections import defaultdict
import google.generativeai as generativeai
from openai import OpenAI
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig
from botocore.config import Config
from dotenv import load_dotenv
from retrieval import *
from prompt import *
import json
import traceback
import time
import random
from utils.token_calculator import tokenize

# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)


PROJECT_ID = "[deepresearch-llm-modeling]"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
GEMINI_MODEL_ID = "gemini-2.5-flash"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.deepseek.r1-v1:0")

_BEDROCK_CONFIG = Config(
    region_name=os.getenv("BEDROCK_REGION", "us-east-2"),
    retries={"max_attempts": 8, "mode": "standard"},
    read_timeout=120,
    connect_timeout=10,
    max_pool_connections=128,
)

_bedrock_client = None

def get_bedrock_client():
    """Return a cached Bedrock runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", config=_BEDROCK_CONFIG)
    return _bedrock_client

ACTIONS = ['search', 'answer', 'plan', 'scripts', 'summary']
CONCURRENT_NUM = 96
MAX_CONTEXT_LENGTH = 8000

class LLMAgent:
    def __init__(self, config, log_dir: str, answer_dir: str, is_long_report: bool = False, verbose: bool = False, is_qwen: bool = True, is_llama: bool = False, is_bedrock: bool = False, search_engine: str = 'clueweb', url: str = 'http://localhost:8000/v1'):
        self.is_long_report = is_long_report
        self.is_qwen = is_qwen
        self.is_llama = is_llama
        self.is_bedrock = is_bedrock
        if sum(int(flag) for flag in (is_qwen, is_llama, is_bedrock)) > 1:
            raise ValueError('Only one of is_qwen, is_llama, or is_bedrock can be True.')
        if is_qwen or is_llama:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=url,
                timeout=60
            )
            self.model_name = self.client.models.list().data[0].id
        elif is_bedrock:
            self.model_name = BEDROCK_MODEL_ID
            self.client = get_bedrock_client()
        else:
            self.model_name = GEMINI_MODEL_ID
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.search_engine = search_engine
        self.consecutive_search_cnt = {} # Number of consecutive search actions performed for each sample
        self.search_cnt = {} # Number of total search actions performed for each sample
        self.script_cnt = {} # Number of total script actions performed for each sample
        self.summary_cnt = {} # Number of total summary actions performed for each sample
        self.context_cnt = {} # Number of total context length in each turn for each sample
        self.turn_id = {} # Turn ID for each question
        self.summary_history = {} # History of summary actions performed for each sample
        self.need_format_reminder = {} # Whether a format reminder prompt is needed for each sample
        self.config = config
        self.verbose = verbose
        self.log_dir = log_dir
        self.answer_dir = answer_dir
        self.questions = {}
        print(f"#######\nInit LLMAgent with model {self.model_name}, mode: {'long' if self.is_long_report else 'short'}, search: {self.search_engine}\n#######")
        time.sleep(5)


    def run_llm_loop(self, prompt, question_id):
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        trajectory_log = f"{self.log_dir}/trajectory_{question_id}.md"
        trajectory_jsonl_log = f"{self.log_dir}/trajectory_{question_id}.jsonl"
        search_log = f"{self.log_dir}/search_{question_id}.log"
        
        # Clear logs
        with open(trajectory_log, 'w', encoding='utf-8') as f:
            f.write('')
        if self.verbose:
            # with open(search_log, 'w', encoding='utf-8') as f:
            #     f.write('')
            with open(trajectory_jsonl_log, 'w', encoding='utf-8') as f:
                f.write('')

        print(f"Running question {question_id}")

        done = False
        input = prompt
        self.consecutive_search_cnt[question_id] = 0
        self.search_cnt[question_id] = 0
        self.script_cnt[question_id] = 0
        self.summary_cnt[question_id] = 0
        self.context_cnt[question_id] = []
        self.turn_id[question_id] = 0
        self.summary_history[question_id] = ''
        self.need_format_reminder[question_id] = False
        try:
            for step in range(self.config["max_turns"]):
                self.turn_id[question_id] += 1
                print(f"=====turn {self.turn_id[question_id]}======")

                if self.is_qwen:
                    response, action = self._query_qwen(input, question_id)
                elif self.is_llama:
                    response, action = self._query_llama(input, question_id)
                elif self.is_bedrock:
                    response, action = self._query_bedrock(input, question_id, trajectory_log)
                else:
                    response, action = self._query_gemini(input, question_id, trajectory_log)
                # execute actions (search or answer) and get observations
                done, need_update_history, next_obs = self._execute_response(
                    action, self.config["num_docs"], question_id, search_log
                )
                self._record_trajectory(input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id)

                if done:
                    print("=====final response======")
                    break
                input = self._update_input(
                    input, response, next_obs, question_id, need_update_history, prompt
                )
            
            answer = self._compose_final_output(action)
            self._log_result(answer=answer, question_id=question_id)
                
            print(f"Question {question_id} result saved to {self.answer_dir}/result_{question_id}.json\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def run_llm_loop_parallel(self, prompts, questions, question_ids):
        print(f"Running {len(prompts)} questions in parallel with {CONCURRENT_NUM} workers")

        for i, question_id in enumerate(question_ids):
            question = questions[i]
            self.questions[question_id] = question

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [executor.submit(self.run_llm_loop, prompt, question_id) 
                for prompt, question_id in zip(prompts, question_ids)]
            concurrent.futures.wait(futures)

    def _query_qwen(self, prompt, question_id):
        """Query Qwen with action format check. Only return the response with correct format.
        Args:
            prompt: prompt
        Returns:
            response: response with correct format and thought process
        """
        thought = None
        max_try_times = 6
        for _ in range(max_try_times):
            try:
                qwen_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20480,
                )
                
                thought = qwen_response.choices[0].message.reasoning_content
                response = qwen_response.choices[0].message.content
                break

            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")

        action = self._postprocess_response(response) # if format is not correct, action is None

        if thought is not None:
            original_response = f'<think>{thought}</think>\n{response}'
        else:
            original_response = response
        
        context_length = tokenize(prompt) + tokenize(original_response)
        self.context_cnt[question_id].append(context_length)

        return original_response, action
    
    def _query_llama(self, prompt, question_id):
        """Query Qwen with action format check. Only return the response with correct format.
        Args:
            prompt: prompt
        Returns:
            response: response with correct format and thought process
        """
        thought = None
        max_try_times = 6
        for _ in range(max_try_times):
            try:
                llama_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20480,
                )
                
                response = llama_response.choices[0].message.content
                break

            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")

        action = self._postprocess_response(response) # if format is not correct, action is None

        original_response = response
        
        context_length = tokenize(prompt) + tokenize(original_response)
        self.context_cnt[question_id].append(context_length)

        return original_response, action

    def _query_bedrock(self, prompt, question_id, trajectory_log):
        """Query an AWS Bedrock model with action format validation.
        Args:
            prompt: prompt
        Returns:
            response: response with correct format and thought process
        """
        max_try_times = 5
        response = None
        thought = ""
        action = None

        for _ in range(max_try_times):
            try:
                bedrock_response = self.client.converse(
                    modelId=self.model_name,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "maxTokens": 25600,
                        "temperature": 0.0,
                    },
                )
                content = bedrock_response.get("output", {}).get("message", {}).get("content", [])
                for part in content:
                    if response is None and "text" in part:
                        response = part["text"]
                    reasoning = part.get("reasoningContent")
                    if reasoning and not thought:
                        reasoning_text = reasoning.get("reasoningText", {})
                        thought = reasoning_text.get("text", thought)

                if response is not None:
                    action = self._postprocess_response(response)
                    if action is not None:
                        break

            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
                else:
                    time.sleep(random.randint(1, 3))

        original_response = f'<think>{thought}</think>\n{response}'

        context_length = tokenize(prompt) + tokenize(original_response)
        self.context_cnt[question_id].append(context_length)

        return original_response, action

    def _query_gemini(self, prompt, question_id, trajectory_log):
        """Query Gemini with action format check. Only return the response with correct format.
        Args:
            prompt: prompt
        Returns:
            response: response with correct format and thought process
        """
        max_try_times = 5
        response = None  
        thought = ""
        action = None
        
        for _ in range(max_try_times):
            try:
                gemini_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerateContentConfig(
                        thinking_config=ThinkingConfig(include_thoughts=True),
                        max_output_tokens=6144,
                        temperature=1.0
                    ),
                )
                for part in gemini_response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if part.thought:
                        thought = part.text
                    else:
                        response = part.text
                
                if response is not None:
                    action = self._postprocess_response(response)
                    if action is not None: # if format is correct, break
                        break
                
            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
                else:
                    # random sleep 1-3 seconds
                    time.sleep(random.randint(1, 3))
           
        original_response = f'<think>{thought}</think>\n{response}'

        context_length = tokenize(prompt) + tokenize(original_response)
        self.context_cnt[question_id].append(context_length)

        return original_response, action

    def _postprocess_response(self, response):
        """Make sure the response is in the correct format.
        Args:
            response: response text
        Returns:
            processed response, if the format is not correct, return None
        """
        if response is None:
            return None
        
        # Count occurrences of each tag
        tag_counts = {}
        for action in ACTIONS:
            start_tag = f'<{action}>'
            end_tag = f'</{action}>'
            start_count = response.count(start_tag)
            end_count = response.count(end_tag)
            tag_counts[action] = {'start': start_count, 'end': end_count}
        
        # no summary action involved, normal case
        if tag_counts['summary']['start'] == 0:
            # Validate tag format rules
            
            # check for <information> or </information> tag, this should not appear in the response
            if '<information>' in response or '</information>' in response:
                return None
            
            valid_actions = []
            for action in ACTIONS:
                start_count = tag_counts[action]['start']
                end_count = tag_counts[action]['end']
                
                # Tags must appear in pairs and at most once
                if start_count != end_count or start_count > 1:
                    return None
                
                # If this action's tags appeared once, record as valid action
                if start_count == 1:
                    valid_actions.append(action)
            
            # Only one action is allowed per response
            if len(valid_actions) != 1:
                return None
            
            # Extract content between valid action tags
            action = valid_actions[0]
            pattern = f'<{action}>(.*?)</{action}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                return f'<{action}>{content}</{action}>'
                
        # special case for summary action, because the content in summary contains other tags
        else: 
            # begin tag and end tag should only appear once
            if tag_counts['summary']['start'] != 1 or tag_counts['summary']['end'] != 1:
                return None
            
            # Find the first occurrence of <summary>
            start_idx = response.find('<summary>')
            # Find the last occurrence of </summary>
            end_idx = response.rfind('</summary>')
            
            # start_idx should be at the beginning of the response, and end_idx should be at the end of the response
            if start_idx != 0 or end_idx != len(response) - len('</summary>'):
                return None  
            
            # Extract content between the first <summary> and last </summary>
            content = response[start_idx + len('<summary>'):end_idx].strip()
            return f'<summary>{content}</summary>'

        
        return None

    def _execute_response(self, action, num_docs, question_id, search_log, do_search=True):
        """
        Args:
            action: action to be executed, None if format is not correct
            num_docs: number of docs to retrieve
            search_log: file to log search output
            do_search: whether to perform search
        Returns:
            done: whether the task is done
            need_update_history: whether need to update the history to agent summary
            next_obs: next observation
        """

        if action is None:
            self.need_format_reminder[question_id] = True
            next_obs = 'A invalid action, cannot be executed.'
            return False, False, next_obs

        action, content = self._parse_action(action)
        next_obs = ''
        done = False
        need_update_history = False

        search_query = content if action == 'search' else ''
        
        if do_search and search_query != '':    
            search_results = self._search(search_query, num_docs, search_log, question_id)
        else:
            search_results = ''

        if action == "answer":
            done = True
            next_obs = f'answer generated, the process is done.'
        elif action == 'search':
            self.search_cnt[question_id] += 1
            self.consecutive_search_cnt[question_id] += 1
            observation = f'<information>{search_results}</information>'
            next_obs = observation
        elif action == 'plan':
            self.consecutive_search_cnt[question_id] = 0
        elif action == 'scripts':
            self.consecutive_search_cnt[question_id] = 0
            self.script_cnt[question_id] += 1
        elif action == 'summary':
            next_obs = 'You performed a summary action in this turn. The content of this action is ignored since your history turns information has been updated according to it.\n'
            self.consecutive_search_cnt[question_id] = 0
            self.summary_cnt[question_id] += 1
            self.summary_history[question_id] = content
            need_update_history = True
        else:
            raise ValueError(f"Invalid action: {action}")

        return done, need_update_history, next_obs

    def _parse_action(self, action):
        """Parse the action to get the action type and content.
        Args:
            action: action, format ensured by postprocess_response
        Returns:
            action_type: action type
            content: action content
        """
        # Find the first occurrence of '<' and '>' to extract action_type
        start_tag_open = action.find('<')
        start_tag_close = action.find('>', start_tag_open)
        if start_tag_open == -1 or start_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")
        
        action_type = action[start_tag_open + 1:start_tag_close]

        # Find the last occurrence of '</' and '>' to locate the closing tag
        end_tag_open = action.rfind('</')
        end_tag_close = action.rfind('>', end_tag_open)
        if end_tag_open == -1 or end_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")

        # Extract content between the first '>' and last '</'
        content = action[start_tag_close + 1:end_tag_open].strip()

        return action_type, content

    def _record_trajectory(self, input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id):
        """Record the trajectory of the agent.
        Args:
            input: input
            response: response
            trajectory_log: path to trajectory log file
        """
        with open(trajectory_log, 'a', encoding='utf-8') as f:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"## Turn {self.turn_id[question_id]} {time}\n\n")

            input_length = tokenize(input)
            response_length = tokenize(response)            
            
            # Create patterns for all action types and truncate long contents
            for action in ['search', 'answer', 'plan', 'scripts', 'information']:
                pattern = f'<{action}>(.*?)</{action}>'
                
                def truncate_action_content(match):
                    """Truncate action content if it's too long"""
                    full_content = match.group(1)  # Content between action tags
                    if len(full_content) > 100:
                        truncated_content = full_content[:100] + '...'
                        return f'<{action}>{truncated_content}</{action}>'
                    else:
                        return match.group(0)  # Return original if short enough
                
                input_short = re.sub(pattern, truncate_action_content, input, flags=re.DOTALL)
            
            f.write(f"### Input:\n**length={input_length}**\n{input_short}\n\n")
            f.write(f"### Response:\n**length={response_length}**\n{response}\n\n--------------------------------\n\n")

        if self.verbose:
            with open(trajectory_jsonl_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "input": input,
                    "response": response,
                    "next_obs": next_obs,
                    "context_length": input_length + response_length
                }) + '\n')

    def _update_input(self, input, cur_response, next_obs, question_id, need_update_history, original_prompt):
        """Update the input with the history.
        Args:
            input: input
            cur_response: current full response
            next_obs: next observation
            need_update_history: whether update the history to agent summary
            original_prompt: original prompt for the question
        Returns:
            updated input
        """
        if self.need_format_reminder[question_id]: # there is no valid action in this turn, need format reminder prompt
            context = f"[Turn {self.turn_id[question_id]}]:\n{cur_response}\n\n"
            context += format_reminder_prompt
            new_input = input + context
            self.need_format_reminder[question_id] = False
        else:
            if need_update_history:
                context = f"[Turn 0 - Turn {self.turn_id[question_id] - 1}]:\n{self.summary_history[question_id]}\n\n"
                context += f"[Turn {self.turn_id[question_id]}]:\n{next_obs}\n\n"
                new_input = original_prompt + context
            else:
                context = f"[Turn {self.turn_id[question_id]}]:\n{cur_response}\n{next_obs}\n\n"
                new_input = input + context

        # add reminder for search and final report
        if self.consecutive_search_cnt[question_id] > self.config["search_reminder_turn"]:
            new_input += f'\nNote: You have performed {self.consecutive_search_cnt[question_id]} search actions. Please consider update your report scripts or output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
        if self.turn_id[question_id] > self.config["final_report_reminder_turn"]:
            new_input += f'\nNote: You have performed {self.turn_id[question_id] + 1} turns. Please consider output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
        
        # add summary reminder prompt if context is too long
        input_length = tokenize(new_input)
        if input_length > MAX_CONTEXT_LENGTH:
            new_input += summary_reminder_prompt

        return new_input

    def _compose_final_output(self, response):
        if response is not None and '<answer>' in response and '</answer>' in response:
            return response.split('<answer>')[1].split('</answer>')[0]
        else:
            return 'did not find answer'

    def _log_result(self, answer, question_id):
        answer_file = f"{self.answer_dir}/result_{question_id}.json"
        with open(answer_file, 'w', encoding='utf-8') as f:
            result = {
                    "model": self.model_name,
                    "question": self.questions[question_id],
                    "answer": answer,
                    "turns": self.turn_id[question_id],
                    "search count": self.search_cnt[question_id],
                    "script count": self.script_cnt[question_id],
                    "summary count": self.summary_cnt[question_id],
                    "context lengths": self.context_cnt[question_id]
                }
            json.dump(result, f, indent=4)

    def _search(self, query, num_docs, search_log, question_id):
        
        if self.search_engine == 'clueweb':
            documents = query_clueweb(query, num_docs=num_docs)
        elif self.search_engine == 'tavily':
            documents = query_tavility(query)
        elif self.search_engine == 'serper':
            documents = query_serper(query)
        elif self.search_engine == 'fineweb':
            documents = query_fineweb(query, num_docs=num_docs)
        else:
            raise ValueError(f"Invalid search engine: {self.search_engine}")
        info_retrieved = "\n\n".join(documents)

        if self.verbose:
            with open(search_log, 'a', encoding='utf-8') as f:
                f.write(f"[turn={self.turn_id[question_id]}]\n")
                f.write(f"query:\n{query}\n\n")
                f.write(f"info_retrieved:\n{info_retrieved}\n\n\n")
        return info_retrieved

def load_questions_from_file(file_path):
    """Load questions from a JSON file (structured as an array of objects with 'id' and 'question' fields)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    ids = [item["id"] for item in data]
    return questions, ids

def filter_completed_questions(questions, ids, answer_dir):
    """Filter out questions that already have answer files"""
    filtered_questions_dict = {}
    completed_count = 0
    
    for i, question_id in enumerate(ids):
        answer_file = f"{answer_dir}/result_{question_id}.json"
        if os.path.exists(answer_file):
            completed_count += 1
        else:
            filtered_questions_dict[question_id] = questions[i]
    
    return filtered_questions_dict, completed_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_file', type=str, help='Path to json file containing array of id and question fields')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--answer_dir', type=str, default='results', help='Result directory')
    parser.add_argument('--long_report', action='store_true', help='Generate long report (default is short answer)')
    parser.add_argument('--is_qwen', action='store_true', help='Use Qwen model')
    parser.add_argument('--is_llama', action='store_true', help='Use Llama model')
    parser.add_argument('--is_bedrock', action='store_true', help='Use AWS Bedrock model')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1', help='URL to use')
    parser.add_argument('--search_engine', type=str, default='clueweb', help='Search engine to use')
    parser.add_argument('--use_explicit_thinking', action='store_true', help='Whether is is a model with internal thinking. For a not thinking model, we use explicit think prompt to guide the model to think.')
    parser.add_argument('--max_turns', type=int, default=15, help='Max number of turns')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    is_long_report = args.long_report
    answer_dir = args.answer_dir
    log_dir = args.log_dir
    is_qwen = args.is_qwen
    is_llama = args.is_llama
    is_bedrock = args.is_bedrock
    url = args.url
    search_engine = args.search_engine
    use_explicit_thinking = args.use_explicit_thinking
    # make sure answer_dir and log_dir exist
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load questions from file
    questions, ids = load_questions_from_file(args.batch_file)
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {args.batch_file}")
    
    # Filter out completed questions
    filtered_questions_dict, completed_count = filter_completed_questions(questions, ids, answer_dir)
    remaining_questions_num = len(filtered_questions_dict)
    
    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions_num} questions")
    
    # If no questions to process, exit
    if remaining_questions_num == 0:
        print("All questions have been completed!")
        exit(0)

    prompts = []
    remaining_questions = []
    remaining_ids = []
    
    for id, question in filtered_questions_dict.items():
        remaining_questions.append(question)
        remaining_ids.append(id)
        
        if is_long_report:
            prompts.append(report_prompt.format(question=question))
        else:
            if args.use_explicit_thinking:
                prompts.append(short_answer_prompt_explicit_thinking.format(question=question))
            else:
                prompts.append(short_answer_prompt_internal_thinking.format(question=question))

    max_turns = args.max_turns
    config = {
              "max_turns": max_turns, # Max number of turns
              "num_docs": 3, # Number of documents to retrieve
              "search_reminder_turn": 5, # Number of turns to remind the agent to stop searching and revise the report scripts or output the final report (only for long report)
              "final_report_reminder_turn": max_turns - 5 # Number of turns to remind the agent to output the final report (only for long report)
              } 
    
    agent = LLMAgent(config, log_dir=log_dir, answer_dir=answer_dir, is_long_report=is_long_report, verbose=True, is_qwen=is_qwen, is_llama=is_llama, is_bedrock=is_bedrock, search_engine=search_engine, url=url)

    if is_long_report:
        print(f"Generating long report mode...")
    else:
        print(f"Generating short answer mode...")

    agent.run_llm_loop_parallel(prompts, remaining_questions, remaining_ids)