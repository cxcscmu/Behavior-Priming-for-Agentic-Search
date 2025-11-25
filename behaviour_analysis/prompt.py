## Identify Reasoning Behaviors

reason_prompt = """
[Instruction]
You are tasked with analyzing multi-step trajectories of a search agent's two attempts for answering the same question using search tools. One of the attempts correctly answers the question, and another attempt does not. Based on the content, please provide a detailed explanation of why one attempt succeeds and the other fails.
There are two parts in each step of the trajectory:
1. Agent output: The agent's output in this step, consists of it's thinking process and the final action.
2. Environment feedback: The feedback from the environment, including the search results wrapped in <information> and </information> tags when the agent performs a search action in this step.

The agent could perform one of the following actions in each step:
1. <search> query </search>: search the web for information
2. <answer> answer </answer>: output the final answer
3. <summary> important parts of the history turns </summary>: summarize the history turns to keep valuable information for solving the question.

Please analyze the agent's behavior in each step and provide a detailed explanation of why one attempt succeeds and the other fails.

[Question]
{question}

[Trajectory 1]
{trajectory_1}

[Evaluation Results 1]
{evaluation_results_1}

[Trajectory 2]
{trajectory_2}

[Evaluation Results 2]
{evaluation_results_2}

[Your Explanation]
"""

extract_behavior_prompt = """
[Instruction]  
You are an expert in analyzing the behavior of a search agent. This agents calls external search tools iteratively to answer a question. You will be provided with an explanation about a search agent's two attempts to answer the same question. The first attempt correctly answers the question, while the second attempt fails.  

Based on the explanation of why trajectory 1 succeeds while trajectory 2 fails, extract the key reasoning behaviors implied by the explanation that lead to the success of trajectory 1. 

1. Focus on generalizable behaviors and reasoning patterns, rather than specific details like websites, query content, etc.
2. Focus on non-linear reasoning behaviors that can be generalized to other questions, rather than linear, monotonic, trivial behaviors that directly and easily achieve the goal.
3. Summarize and introduce behaviors as beneficial reasoning patterns for multi-step iterative search agents.

Return the list as a JSON array of strings. Do not include markdown code fences. If there are no behavior-like statements, return an empty JSON array.

[Explanation]  
{explanation_text}
"""

merge_prompt = """
[Instruction]
You are an expert in analyzing the behavior of a search agent. You are provided with a set of behaviors describing the the reasoning process and actions of the agent.  

Below is a list of behaviors regarding the behavior of the search agent. Some behaviors may be duplicates or express very similar meanings. Please merge them by removing duplicates and consolidating similar behaviors, while keeping only the most essential information. 
The final behaviors should be clear and objective with some explanation, so they can be reliably used to evaluate the agent's reasoning and interaction trajectory.  

Return the merged list as a JSON array of strings. Do not include markdown code fences.  

[Behaviors]  
{behaviors_text}
"""


# Behavior Frequency Analysis

judge_behavior_prompt = """
[Instruction]
You are tasked with analyzing a multi-step trajectory of a search agent's attempt for answering a question using search tools. 

The agent can perform one of the following actions in each step:
1. <search> query </search>: search the web for information
2. <answer> answer </answer>: output the final answer
3. <summary> important parts of the history turns </summary>: summarize the history turns to keep valuable information for solving the question.

There are two parts in each step of the trajectory:
1. Agent output: The agent's output in this step, consists of it's thinking process and the final action.
2. Environment feedback: The feedback from the environment, including the search results wrapped in <information> and </information> tags when the agent performs a search action in this step.

Please act as an judge to evaluate whether the agent's thinking process and actions in this trajectory demonstrated any of following behaviors:

**behavior1: Information Verification**
The agent validates information across multiple reliable sources to ensure its conclusions are well-founded.
* **Cross-Referencing:** Actively seeking out and comparing multiple sources to confirm critical facts, or performing additional searches to verify the information.
* **Citing Evidence:** Explicitly basing its reasoning and conclusions on the information found, rather than making unsupported claims.

**behavior2: Authority Evaluation**
The agent assesses the reliability of its sources and resolves conflicting information.
* **Detecting Conflicts:** Identifying when different sources provide conflicting information and attempting to resolve the discrepancy.
* **Prioritizing Authority:** Giving more weight to official documentation, academic papers, and reputable news outlets over forums, blogs, or less reliable sources.

**behavior3: Adaptive Search**
The agent intelligently modifies its search strategy based on the information and challenges encountered in previous steps.
* **Narrowing Focus:** Using initial broad search results to identify more specific and effective keywords for subsequent searches.
* **Broadening Scope:** Widening the search terms or approach when initial queries are too narrow and yield no useful results.

**behavior4: Error Recovery**
The agent recognizes previous errors and takes actions to correct its course.
* **Acknowledging Failure:** Explicitly noting when a search query or an entire strategy is not yielding useful information, or some mistakes are made.
* **Strategic Pivoting:** Decisively abandoning a failed approach and formulating a new plan to achieve the user's goal, or taking actions to correct the mistakes.

Be as objective as possible when evaluating the behaviors and do not evaluate other characteristics of the response. If the behavior is not applicable for this task, treat it as if the behavior is not demonstrated.

You must provide your answer with the following json format without markdown code fences:

{{
  "behavior1": "<'Yes' or 'No'>",
  "behavior2": "<'Yes' or 'No'>",
  "behavior3": "<'Yes' or 'No'>",
  "behavior4": "<'Yes' or 'No'>",
}}

[Question]
{question}

[Trajectory]
{trajectory}

[Your Answer]
"""