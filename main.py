from processes import CoAProcessRunner
from chain_of_agents import ChainOfAgents
from chain_of_agents import ChainOfAgents
from chain_of_agents.debug_agents import SingleAgent, ReactAgent
import os
from dotenv import load_dotenv
import pathlib
import sys
import nltk
nltk.download('punkt_tab')
from huggingface_hub import login
# import spacy.cli
# spacy.cli.download("en_core_web_sm")

login("YOUR-HUGGINGFACE-TOKEN") #Jackingchen09
from DebugBench.leetcode_oj.leetcode_tester import LeetCodeTester
from chain_of_agents.prompt import single_prompt, react_prompt, flow_prompt, DF_prompt
from datasets import load_dataset
import json
from tqdm import tqdm
import re
import argparse
import pandas as pd
# Initialize LeetCode tester
tester = LeetCodeTester()

# Get CLI argument from bash
# ["evaluate","reevaluate"] 
arg = "evaluate"

# Load environment variables
env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)



def transform_code(raw_code: str) -> str:
    # Step 1: Decode escape characters like \n, \t
    decoded = raw_code.encode('utf-8').decode('unicode_escape')

    # Step 2: Remove triple-quoted comments (both """ and ''')
    cleaned = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', decoded, flags=re.DOTALL)

    # Step 3: Strip trailing spaces on empty lines and normalize indentation
    # This will keep code readable
    cleaned_lines = [line.rstrip() for line in cleaned.splitlines()]

    return '\n'.join(cleaned_lines).strip()

def evaluate(dataset, max_questions=200, language="python3", prompt_name="flow"):
    total = 0
    id = 0
    total_pass = 0
    exact = 0
    all_results = []
    
    # Create output directory if it doesn't exist
    output_dir = pathlib.Path("Output")
    output_dir.mkdir(exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = output_dir / f"{prompt_name}.json"
    
    # Load existing results if checkpoint exists
    processed_ids = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                existing_results = json.load(f)
                all_results = existing_results
                processed_ids = {result["id"] for result in existing_results}
                total_pass = sum(1 for result in existing_results if result["passed"])
                total = len(existing_results)
                print(f"Loaded checkpoint: {total} questions processed, {total_pass} passed")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load checkpoint file {checkpoint_file}: {e}")
            all_results = []

    total_questions = min(len(dataset), max_questions)

    with tqdm(total=total_questions, desc=f"Evaluating DebugBench") as pbar:
        # Update progress bar to current position
        pbar.update(total)
        
        for entry in dataset:
            if total >= max_questions:
                break
            
            # Skip if already processed
            if id in processed_ids:
                id += 1
                continue

            task_id = entry["slug"]
            question = entry["question"]
            example = entry["examples"]
            constraints = entry["constraints"]
            query = f"Question: {question}\nExample: {example}\nConstraints: {constraints}"
            buggy_code = entry["buggy_code"]
            bug_type = entry.get("subtype", "unknown")
            solution = entry["solution"]
            bug_explanation = entry.get("bug_explanation", "")
            gold_answer = solution

            # Run your agent
            runner = CoAProcessRunner(coa)
            pred, wrong_type = runner.debug_react(query, buggy_code, "", "")
            code = transform_code(pred)
            code = "# DebugBench Solution\n" + code
            
            import time
            MAX_RETRIES = 4
            RETRY_DELAY = 5

            try:
                for attempt in range(MAX_RETRIES):
                    reward, result = tester.test(code, task_id, language)

                    state = result.get("state", "").upper()
                    if state in ["PENDING", "STARTED"]:
                        print(f"[{task_id}] State is {state}. Waiting before retrying... (Attempt {attempt+1}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"result: {reward}")
                        passed = reward
                        error = False
                        break
                else:
                    print(f"[{task_id}] Submission stuck in state {state}. Giving up after {MAX_RETRIES} attempts.")
                    passed = False
                    error = True
                    
            except Exception as e:
                print(f"Submission failed for {task_id}: {e}")
                passed = False
                error = True
                result = {"error": str(e)}

            # Record all results (both passed and failed)
            if not error:
                result_entry = {
                    "slug": task_id,
                    "id": id,
                    "question": question,
                    "buggy_code": buggy_code,
                    "gold_solution": solution,
                    "prediction": pred,
                    "passed": passed,
                    "submission_result": result,
                    "bug_explanation": bug_explanation,
                    "subtype": entry.get("subtype", ""),
                    "level": entry.get("level", ""),
                    "error": False
                }
                all_results.append(result_entry)
                
                if passed:
                    total_pass += 1
                else:
                    print(f"❌ Failed: problem #{id}: {task_id}")
                
                total += 1
                
                # Save checkpoint after each successful processing
                with open(checkpoint_file, "w") as fout:
                    json.dump(all_results, fout, indent=2)
            
            id += 1
            if total > 0:
                print(f"Pass rate now: {total_pass / total:.4f} on {total_pass}/{total}")
            pbar.update(1)

    pass_rate = total_pass / total if total > 0 else 0
    print(f"{language} Pass Rate: {pass_rate:.4f} on {total} examples")
    print(f"Exact: {exact}")

    # Final save
    with open(checkpoint_file, "w") as fout:
        json.dump(all_results, fout, indent=2)
    
    # Also save wrong answers separately for backward compatibility
    wrong_answers = [result for result in all_results if not result["passed"]]
    with open(f"DebugBench_Flow.json", "w") as fout:
        json.dump(wrong_answers, fout, indent=2)

    return pass_rate

parser = argparse.ArgumentParser(description="Select a prompt type to run.")
parser.add_argument(
    "--prompt",
    default='DF',
    help="Choose a prompt type: single, react, flow, DF"
)
args = parser.parse_args()

prompt = args.prompt

Prompts={
    "single": single_prompt,
    "react" : react_prompt,
    "flow"  : flow_prompt,
    "DF"    : DF_prompt
}


# Initialize Chain of Agents
coa = ChainOfAgents(
    worker_model="gpt-4.1-mini",
    manager_model="gpt-4.1-mini",
    action_model="gpt-4.1-mini",
    chunk_size=3000,
    max_new_tokens=70000,
    single_prompt = single_prompt,
    react_prompt  = Prompts[prompt],    
)
# DF_prompt, flow_prompt, react_prompt

# Load account from file 
# account_file = "LCaccounts.xlsx"
# df = read_excel_accounts(account_file)
# for index, row in df.iterrows():
#     username = row.get('Username', '')
#     csrf_token = row.get('LEETCODE_CSRF_TOKEN', '')
#     session_token = row.get('LEETCODE_SESSION', '')
    
#     print(f"\n--- Checking account: {username} ---")
# os.environ['LEETCODE_CSRF_TOKEN'] = csrf_token if csrf_token else ''
# os.environ['LEETCODE_SESSION'] = session_token if session_token else ''

ds = load_dataset("Rtian/DebugBench", split="test")
# filtered_ds = ds.filter(lambda example: example["language"] == "python3" and example["level"] == "hard" )



# VALIDATION
if arg == 'evaluate':
    # evaluate(ds, max_questions=99999, language="python3")
    evaluate(ds, max_questions=99999, language="python3", prompt_name=prompt)
elif arg == 'reevaluate':
    re_evaluate_wrong(ds, language="python3", json_path="DebugBench_python3_single_onlycode.json")
else:
    print(f"Unknown command: {arg}")
    sys.exit(1)

# TEST (optional)
# test_path = "QuALITY.v1.0.1.test"
# evaluate(test_path, "Test")