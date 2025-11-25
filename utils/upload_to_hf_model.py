
from huggingface_hub import HfApi, create_repo
import os
import argparse

def upload_folder_to_huggingface(
    local_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload model folder"
):

    api = HfApi()

    print(f"Checking or creating repository: {repo_id}")
    try:
        api.repo_info(repo_id=repo_id)
        print(f"Repository {repo_id} already exists.")
    except Exception:
        print(f"Repository {repo_id} does not exist, creating...")
        parts = repo_id.split('/')
        if len(parts) == 2:
            organization_or_user, repo_name_only = parts
            create_repo(repo_id=repo_name_only, organization=organization_or_user, private=private, exist_ok=True)
        else: 
            create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Repository {repo_id} created successfully.")

    print(f"Uploading {local_path} to {repo_id}...")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model", # Or "dataset", "space"
        commit_message=commit_message,
        ignore_patterns=[".git", ".DS_Store"], # Ignore common files that don't need to be uploaded
    )
    print(f"Successfully uploaded {local_path} to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub.")
    parser.add_argument("--private", action="store_true", help="Set the repository as private.")
    parser.add_argument("--commit_message", type=str, default="Upload model folder", help="Commit message for the upload.")

    args = parser.parse_args()

    local_path = "/data/group_data/cx_group/verl_agent_shared/checkpoint/apm_sft_llama_3.2_3b_instruct/"
    repo_id = "zizi-0123/apm_sft_llama_3.2_3b_instruct"

    upload_folder_to_huggingface(
        local_path=local_path,
        repo_id=repo_id,
        private=args.private,
        commit_message=args.commit_message
    )
