from typing import List, Optional, Iterator, Dict
from openai import OpenAI
import os
import requests
import re
import time
from openai._exceptions import RateLimitError, APIError

class SingleAgent:
    """Worker agent that processes individual chunks of text using Azure OpenAI."""

    def __init__(self, deployment: str, system_prompt: str="""
You are a Single Agent first try to generate a reference code based on the problem description ONLY. 
Then, given a buggy code, modify it until it becomes functionally equivalent to the reference code.
The full flow:
1. imagine some test cases based on the problem description. INCLUDING the corner cases.
2. generate the reference code
3. explain the logic in the reference code
4. analyze the buggy code
5. test all of the test cases on the buggy code and the reference code to make sure they output the same result. Show the outputs from the reference code and the buggy code.
6. explain the logic difference between two codes.
7. modify the buggy code to the reference code.

Output ONLY the final corrected Python3 code. Do not include explanations or extra text. No markdown code blocks.

Modify Constraints:
- Do not alter problem requirements.
- Only change the buggy portions of the buggy code.
- Keep the original coding style intact.
""", max_new_tokens: int = 3000):
        """
        Initialize a worker agent.
        
        Args:
            deployment: The Azure deployment name (e.g., "gpt-35-turbo")
            system_prompt: The system prompt that defines the worker's role
            max_new_tokens: Maximum tokens to generate
        """
        self.deployment = deployment
        self.system_prompt = system_prompt
        self.api_key = os.getenv("OPENAI_API_KEY_GPT4o")  # Or a separate one if needed
        self.endpoint = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment}/chat/completions"
        if deployment in ['gpt-4o-mini','gpt-4.1-mini']:
            self.api_version = "2025-01-01-preview"  # Adjust to your API version
        else:
            self.api_version = "2024-08-01-preview"  # Adjust to your API version
        self.max_new_tokens = max_new_tokens

    def synthesize(self, query: str, code: str, bug_type: str, explanation: str) -> str:
        """
        Process a single chunk of text.
        
        Args:
            chunk: The text chunk to process
            query: The user's query
            previous_cu: The previous Cognitive Unit (CU) if any
            
        Returns:
            str: The processed output for this chunk
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Problem: {query}\n\nBuggy Code: {code}\n\n"
            )}
        ]

        url = f"{self.endpoint}?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        body = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_new_tokens
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    
