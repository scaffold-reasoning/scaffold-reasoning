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
# import spacy.cli
# spacy.cli.download("en_core_web_sm")

# from DebugBench.leetcode_oj.leetcode_tester import LeetCodeTester
from chain_of_agents.prompt import SR_prompt
from datasets import load_dataset
import json
from tqdm import tqdm
import re
import argparse
import pandas as pd

import fcntl
import time
import json
import pathlib
from contextlib import contextmanager
from datasets import load_from_disk

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time
# Get CLI argument from bash
# ["evaluate","reevaluate"] 
arg = "evaluate"

# Load environment variables
env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class LocalLLMSynthesizer:
    def __init__(self, model_name="codellama/CodeLlama-7b-Instruct-hf", device="cuda", max_length=2048):
        self.device = device
        self.max_length = max_length
        self.model_name = model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if safetensors files exist
        import glob
        safetensors_files = glob.glob(os.path.join(model_name, "*.safetensors"))
        use_safetensors = len(safetensors_files) > 0
        
        print(f"Loading model from: {model_name}")
        print(f"Using safetensors: {use_safetensors}")
        
        # Load model with optimizations for large models
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=use_safetensors
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Trying with weights_only=False (less secure)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=False
            )
    
    def format_prompt(self, system_prompt, user_content):
        """Format prompt based on model type"""
        # Qwen (ChatML style)
        if "qwen" in self.model_name.lower():
            # Official Qwen uses ChatML tokens
            return (
                "<|im_start|>system\n"
                f"{system_prompt}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{user_content}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        # DeepSeek-Coder format
        elif "deepseek" in self.model_name.lower():
            print(f"using deepseek style prompt")
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_content}\n<|assistant|>\n"
        
        # Llama-2-chat format
        elif "llama-2" in self.model_name.lower() and "chat" in self.model_name.lower():
            print(f"using llama-2 style prompt")
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"
        
        # WizardCoder format
        elif "wizard" in self.model_name.lower():
            print(f"using wizard style prompt")
            return f"{system_prompt}\n\n### Instruction:\n{user_content}\n\n### Response:\n"
        
        # CodeLlama-Instruct format (your current default)
        elif "codellama" in self.model_name.lower() and "instruct" in self.model_name.lower():
            print(f"using codellama style prompt")
            return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"
        
        # Generic format for other models
        else:
            print("using non of the known models")
            return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"
        
    def synthesize_single(self, user_prompt, query, code, bug_type, explanation, verbose=False):
        """Process a single request"""
        system_prompt="You are a helpful coding assistant."
        # Format according to the message structure you specified
        user_content = f"{user_prompt}\n Problem: {query}\n\nBuggy Code: {code}\n\n"
        
        # Use model-specific prompt formatting
        full_prompt = self.format_prompt(system_prompt, user_content)
        if verbose:
            print(f"The full prompt:\n {full_prompt}")
        # Tokenize the prompt
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False  # prevent creation when supported
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()



def transform_code(raw_code: str) -> str:
    # Step 1: Decode escape characters like \n, \t
    decoded = raw_code.encode('utf-8').decode('unicode_escape')

    # Step 2: Remove triple-quoted comments (both """ and ''')
    cleaned = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', decoded, flags=re.DOTALL)

    # Step 3: Strip trailing spaces on empty lines and normalize indentation
    # This will keep code readable
    cleaned_lines = [line.rstrip() for line in cleaned.splitlines()]

    return '\n'.join(cleaned_lines).strip()

def synthesize(query, code, bug_type, explanation, user_prompt, model_name):
    """
    Modified synthesize function to use local LLM for single requests
    Follows the message structure: system_prompt + user content
    """
    global local_llm_synthesizer
    
    # Initialize the synthesizer if it doesn't exist
    if local_llm_synthesizer is None:
        # Replace with your desired model
        # model_name = "/home/jackcp/research/chain-of-agent/aic-nas2/llm/llms_download/models"  # or any other model
        print(f"Loading model: {model_name}")
        local_llm_synthesizer = LocalLLMSynthesizer(model_name)
        print("Model loaded successfully!")
    
    # Process single request
    return local_llm_synthesizer.synthesize_single(user_prompt, query, code, bug_type, explanation)

    
def evaluate(dataset, max_questions=200, prompt_name="flow", run_stage1=True, run_stage2=True, output_dir="Output_Withstage", gen_mdl="codellama_CodeLlama-7b-Instruct-hf", model_name='/home/jackcp/aic-nas2/llm/llms_download/models/codellama_CodeLlama-7b-Instruct-hf'):
    total = 0
    id = 0
    total_pass = 0
    exact = 0
    all_results = []
    

    # Define checkpoint file path
    checkpoint_file =  f"{output_dir}/{gen_mdl}__{prompt_name}.json"
    
    # Load existing results if checkpoint exists
    processed_entries = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                existing_results = json.load(f)
                all_results = existing_results
                processed_entries = {result["id"]: result for result in existing_results}
                total_pass = sum(1 for result in existing_results if result.get("passed", False))
                total = len(existing_results)
                print(f"Loaded checkpoint: {total} questions processed, {total_pass} passed")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load checkpoint file {checkpoint_file}: {e}")
            all_results = []

    total_questions = min(len(dataset), max_questions)

    with tqdm(total=total_questions, desc=f"Evaluating DebugBench") as pbar:
        # Update progress bar to current position
        pbar.update(len([r for r in all_results if r["id"] < total_questions]))
        
        for entry in dataset:
            if id >= max_questions:
                break

            task_id = entry["slug"]
            question = entry["question"]
            example = entry["examples"]
            constraints = entry["constraints"]
            LANGUAGE = entry["language"]
            query = f"Question: {question}\nExample: {example}\nConstraints: {constraints}"
            buggy_code = entry["buggy_code"]
            bug_type = entry.get("subtype", "unknown")
            solution = entry["solution"]
            bug_explanation = entry.get("bug_explanation", "")
            gold_answer = solution

            # Check existing entry
            existing_entry = processed_entries.get(id)
            
            # Determine what stages to run
            need_stage1 = run_stage1 and (not existing_entry or not existing_entry.get("prediction"))
            need_stage2 = run_stage2 and (not existing_entry or existing_entry.get("passed") is None)
            
            # Skip if nothing needs to be done
            if not need_stage1 and not need_stage2:
                id += 1
                continue

            # Initialize result entry
            if existing_entry:
                result_entry = existing_entry.copy()
            else:
                result_entry = {
                    "slug": task_id,
                    "id": id,
                    "question": question,
                    "buggy_code": buggy_code,
                    "gold_solution": solution,
                    "prediction": "",
                    "code": "",
                    "passed": None,
                    "submission_result": {},
                    "bug_explanation": bug_explanation,
                    "subtype": entry.get("subtype", ""),
                    "level": entry.get("level", ""),
                    "error": False
                }

            # Stage 1: LLM predict code
            if need_stage1:
                import time
                start_time = time.perf_counter()
                print(f"Running Stage 1 for problem #{id}: {task_id}")
                
                # Get the system prompt from Prompts[prompt]
                user_prompt = Prompts[prompt]
                
                # Call synthesize with the system prompt
                pred = synthesize(query, buggy_code, bug_type, bug_explanation, user_prompt, model_name)

                code = transform_code(pred)
                if LANGUAGE.lower() == "python3":
                    code = "# DebugBench Solution\n" + code
                elif LANGUAGE.lower() == "cpp":
                    code = "// DebugBench Solution\n" + code
                elif LANGUAGE.lower() == "java":
                    code = "// DebugBench Solution\n" + code

                end_time = time.perf_counter()  # ⏱️ End measuring
                runtime = end_time - start_time
                # Update result entry with stage 1 results
                result_entry["prediction"] = pred
                result_entry["code"] = code
                result_entry["runtime"] = runtime

                
                # Save checkpoint after stage 1
                if existing_entry:
                    # Update existing entry in all_results
                    for i, result in enumerate(all_results):
                        if result["id"] == id:
                            all_results[i] = result_entry
                            break
                else:
                    # Add new entry
                    all_results.append(result_entry)
                
                with open(checkpoint_file, "w") as fout:
                    json.dump(all_results, fout, indent=2)
                print(f"Stage 1 completed and saved for problem #{id}")

            # Stage 2: Send to LeetCode API
            if need_stage2 and result_entry.get("code"):
                print(f"Running Stage 2 for problem #{id}: {task_id}")
                code = result_entry["code"]
                
                import time
                MAX_RETRIES = 4
                RETRY_DELAY = 5

                try:
                    for attempt in range(MAX_RETRIES):
                        reward, result = tester.test(code, task_id, LANGUAGE)

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

                # Update result entry with stage 2 results
                result_entry["passed"] = passed
                result_entry["submission_result"] = result
                result_entry["error"] = error
                
                # Update the entry in all_results
                for i, res in enumerate(all_results):
                    if res["id"] == id:
                        all_results[i] = result_entry
                        break
                
                # Save checkpoint after stage 2
                with open(checkpoint_file, "w") as fout:
                    print(f"update No.{id},  result_entry: {result_entry} to checkpoint_file: {checkpoint_file}")
                    json.dump(all_results, fout, indent=2)
                
                if passed:
                    total_pass += 1
                    print(f"✅ Passed: problem #{id}: {task_id}")
                else:
                    print(f"❌ Failed: problem #{id}: {task_id}")
                
                print(f"Stage 2 completed and saved for problem #{id}")

            # Update counters
            if result_entry.get("passed") is not None:
                if id not in [r["id"] for r in all_results[:-1] if r.get("passed") is not None]:
                    total += 1
            
            id += 1
            if total > 0:
                current_pass = sum(1 for r in all_results if r.get("passed", False))
                print(f"Pass rate now: {current_pass / total:.4f} on {current_pass}/{total}")
            pbar.update(1)

    # Calculate final statistics
    completed_results = [r for r in all_results if r.get("passed") is not None]
    total_completed = len(completed_results)
    total_passed = sum(1 for r in completed_results if r["passed"])
    
    pass_rate = total_passed / total_completed if total_completed > 0 else 0
    print(f"Total passed: {total_passed}")

    # Final save
    with open(checkpoint_file, "w") as fout:
        json.dump(all_results, fout, indent=2)
    
    # Also save wrong answers separately for backward compatibility
    wrong_answers = [result for result in all_results if result.get("passed") == False]
    with open(f"DebugBench_Flow.json", "w") as fout:
        json.dump(wrong_answers, fout, indent=2)

    
    return pass_rate


def parse_arguments():
    parser = argparse.ArgumentParser(description='Debug Bench with stage control')
    parser.add_argument('--prompt', type=str, default='DF',
                       help='Prompt type to use')
    parser.add_argument('--run-stage1', action='store_true', default=True,
                       help='Run stage 1 (LLM prediction)')
    parser.add_argument('--no-stage1', action='store_true', default=False,
                       help='Skip stage 1 (LLM prediction)')
    parser.add_argument('--run-stage2', action='store_true', default=False,
                       help='Run stage 2 (LeetCode API testing)')
    parser.add_argument('--no-stage2', action='store_true', default=True,
                       help='Skip stage 2 (LeetCode API testing)')
    parser.add_argument('--max-questions', type=int, default=1500,
                       help='Maximum number of questions to process')
    # parser.add_argument('--language', type=str, default='python3',
    #                    help='Programming language for LeetCode submission')
    parser.add_argument('--model-path', type=str, 
                       default='/home/jackcp/aic-nas2/llm/llms_download/models/codellama_CodeLlama-7b-Instruct-hf',
                       help='Path to the model directory')
    parser.add_argument('--output-dir', type=str, default='Output_Withstage',
                       help='Output directory for results')
    return parser.parse_args()

# Global synthesizer instance
local_llm_synthesizer = None
    
# Update the main execution section
if __name__ == "__main__":
    args = parse_arguments()
    prompt = args.prompt
    model_name = args.model_path
    output_dir = args.output_dir
    
    # Prompts={
    #     "SR"    : SR_prompt,     
    # }

    
    

    
    # Initialize LeetCode tester
    # tester = LeetCodeTester()
    
    # Handle stage control logic
    run_stage1 = args.run_stage1 and not args.no_stage1
    run_stage2 = args.run_stage2 and not args.no_stage2
    
    # Validate that at least one stage is enabled
    if not run_stage1 and not run_stage2:
        print("Error: At least one stage must be enabled. Use --run-stage1 or --run-stage2")
        exit(1)
    # Initialize Chain of Agents
    """
    coa = ChainOfAgents(
        model="gpt-4.1-mini",
        chunk_size=3000,
        max_new_tokens=70000,
        prompt  = Prompts[prompt],    
    )
    """

    # ds = load_dataset("Rtian/DebugBench", split="test")
    
    ds = load_from_disk("debugbench")
    filtered_ds = ds.filter(lambda example: example["language"] == "python3" )

    print(f"setting args.max_questions to {args.max_questions}\n ")
    print(f"model_name: {model_name}\n")
    print(f"gen_mdl: {os.path.basename(model_name)}\n")
    # VALIDATION
    evaluate(filtered_ds, 
            max_questions=args.max_questions, 
            prompt_name=prompt,
            run_stage1=run_stage1,
            run_stage2=run_stage2,
            model_name=model_name,
            gen_mdl=os.path.basename(model_name),
            )
