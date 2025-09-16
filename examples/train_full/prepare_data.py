#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SFT JSON + ZIP, update dataset_info.json, unzip videos.
Usage:
    python prepare_data.py --sft_data_name vs2_sft_17k
"""

import os
import shutil
import json
import argparse
import requests
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile

HF_DATASET_USER = "hexuan21"
HF_DATASET_NAME = "vs2_sft_data"
HF_VIDEO_REPO   = "hexuan21/vs2_sft_video"
CHUNK = 1 << 14  # 16 KB


def download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")


def update_dataset_info(sft_data_name: str, data_file: str, ds_info_path: str = "data/dataset_info.json", frame_or_video: str = "video"):
    os.makedirs(os.path.dirname(ds_info_path), exist_ok=True)

    if os.path.exists(ds_info_path):
        with open(ds_info_path, "r", encoding="utf-8") as f:
            ds_infos = json.load(f)
    else:
        ds_infos = {}
    if frame_or_video in ["videos","video","v"]:
        ds_infos[sft_data_name] = {
            "file_name": data_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "videos": "videos"
            }
        }
        
    else:
        ds_infos[sft_data_name] = {
            "file_name": data_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            }
        }
    with open(ds_info_path, "w", encoding="utf-8") as f:
        json.dump(ds_infos, f, indent=2, ensure_ascii=False)
    print(f"[ok] dataset_info.json updated → {ds_info_path}")


def main(sft_data_name: str, frame_or_video: str):
    data_file  = f"{sft_data_name}.json"
    data_save = os.path.join("data", data_file)
    core_name=sft_data_name
    if "no_cot" in core_name:
        core_name=sft_data_name.replace("_no_cot","")
    
    from huggingface_hub import list_repo_files
    hf_video_repo_files = list_repo_files(HF_VIDEO_REPO,repo_type="dataset")
    
    if frame_or_video in ["frames","frame","f"]:
        zip_file_list=[f for f in hf_video_repo_files if core_name in f and f.endswith(".zip") and "frames" in f]
        f_v_dir = os.path.join("data", "frames")
        
    if frame_or_video in ["videos","video","v"]:
        zip_file_list=[f for f in hf_video_repo_files if core_name in f and f.endswith(".zip") and "videos" in f]
        f_v_dir = os.path.join("data", "videos")
    print(zip_file_list)
    
    data_url = f"https://huggingface.co/datasets/{HF_DATASET_USER}/{HF_DATASET_NAME}/resolve/main/{data_file}"
    download_file(data_url, data_save, overwrite=True)
    with open(data_save, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(data_save, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    update_dataset_info(sft_data_name, data_file)
    
    
    # for zip_file in zip_file_list:
    #     zip_url = f"https://huggingface.co/datasets/{HF_VIDEO_REPO}/resolve/main/{zip_file}"
    #     zip_save = os.path.join("data", f"{zip_file}")
    #     download_file(zip_url, zip_save, overwrite=True)
    #     os.makedirs(f_v_dir, exist_ok=True)
    #     try:
    #         with ZipFile(zip_save) as zf:
    #             zf.extractall(f_v_dir)
    #         print(f"[ok] Unzipped → {f_v_dir}")
    #     except BadZipFile as e:
    #         print(f"[error] Bad zip file: {e}")
    #     os.remove(zip_save)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT JSON & ZIP")
    parser.add_argument("--sft_data_name", required=True)
    parser.add_argument("--frame_or_video", required=True)
    args = parser.parse_args()
    main(args.sft_data_name, args.frame_or_video)
