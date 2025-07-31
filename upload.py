import os
from huggingface_hub import HfApi, upload_file
import argparse

def upload_files_individually_to_hub(
    local_dir: str,
):
    token=os.environ["HF_TOKEN"]
    model_name=local_dir.split('/')[-1]
    repo_id=f"videoscore2/{model_name}"
    repo_type="model"
    
    api = HfApi(token=token)
    
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
        
    for file in os.listdir(local_dir):
        abs_path = os.path.join(local_dir, file)
        if os.path.isfile(abs_path):  # skip subdirectories
            print(f"📤 Uploading {file} ...")
            upload_file(
                path_or_fileobj=abs_path,
                path_in_repo=file,  # flat structure
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=f"Upload {file}"
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
