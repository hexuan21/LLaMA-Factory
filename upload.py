import os
from huggingface_hub import HfApi, upload_file
import argparse

def upload_files_individually_to_hub(
    local_dir: str,
):
    token=os.environ["HF_TOKEN"]
    model_name=local_dir.split('/')[-1]
    repo_id=f"videoscore2/{model_name}"
    
    api = HfApi(token=token)
    
    
    repo_type="model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✅ Repository '{repo_id}' already exists.")
    except Exception as e:
        print(f"🚧 Repository '{repo_id}' not found. Creating a new public repo...")
        api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            private=False
        )
    if not os.path.exists(local_dir):
        print(f"🚧 local dir not exist.")
        
    for root, _, files in os.walk(local_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            relative_path = os.path.relpath(abs_path, local_dir)  # path inside repo
            print(f"Uploading {relative_path} ...")

            upload_file(
                path_or_fileobj=abs_path,
                path_in_repo=relative_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=f"Upload {relative_path}"
            )

    print(f"✅ All files in '{local_dir}' have been uploaded to {repo_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, required=True, help="Local folder path to upload")
    args = parser.parse_args()
    print(args.local_dir)
    upload_files_individually_to_hub(
        local_dir=args.local_dir,
    )
