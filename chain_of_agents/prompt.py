# ======================== Scaffold reasoning prompt ========================
SR_prompt="""
You are a Single Agent first try to generate a reference code based on the problem description ONLY. 
Then, given a buggy code, modify it until it becomes functionally equivalent to the reference code.
The full flow:
1. imagine some test cases based on the problem description. INCLUDING the corner cases.
2. generate the reference from the problem description ONLY.
3. explain the logic in the reference code
4. analyze the buggy code
5. test all of the test cases on the buggy code and the reference code to make sure they output the same result. Show the outputs from the reference code and the buggy code.
6. explain the logic difference between two codes.
7. consider both two codes, modify the buggy code to a correct one.

Output ONLY the final corrected Python3 code. Do not include explanations or extra text. No markdown code blocks.

Modify Constraints:
- Do not alter problem requirements.
- Only change the buggy portions of the buggy code.
- Keep the original coding style intact.
"""

