from huggingface_hub import login, hf_hub_download
import os

# ---------------------------
# 1. Hugging Face token
# ---------------------------
# Replace with your Hugging Face token
HF_TOKEN = "TOKEN"

# Log in programmatically
login(token=HF_TOKEN)

# ---------------------------
# 2. Model repo and local folder
# ---------------------------
model_repo = "Qwen/Qwen2.5-Coder-3B-Instruct"  # HF repo ID
local_dir = "./models/Qwen_Qwen2.5-Coder-3B-Instruct"
os.makedirs(local_dir, exist_ok=True)

# ---------------------------
# 3. Download model files
# ---------------------------

# Download config.json
config_path = hf_hub_download(repo_id=model_repo, filename="config.json", cache_dir=local_dir)
print(f"Downloaded config to {config_path}")

# Download tokenizer.json
tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.json", cache_dir=local_dir)
print(f"Downloaded tokenizer to {tokenizer_path}")

# Example: download first model shard (update filenames if needed)
try:
    model_shard = hf_hub_download(repo_id=model_repo, filename="pytorch_model-00001-of-00002.bin", cache_dir=local_dir)
    print(f"Downloaded model shard to {model_shard}")
except:
    print("Check the exact model shard filenames in the HF repo. Some models have multiple shards.")

print("All files downloaded to:", local_dir)
