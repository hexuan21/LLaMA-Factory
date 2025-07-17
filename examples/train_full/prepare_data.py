#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SFT JSON + ZIP, update dataset_info.json, unzip videos.
Usage:
    python prepare_data.py --sft_data_name vs2_sft_17k
"""

import os
import json
import argparse
import requests
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile

HF_DATASET_USER = "hexuan21"
HF_DATASET_NAME = "vs2_sft"
HF_VIDEO_NAME   = "vs2_sft_video"
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


def update_dataset_info(sft_data_name: str, data_file: str, ds_info_path: str = "data/dataset_info.json"):
    os.makedirs(os.path.dirname(ds_info_path), exist_ok=True)

    if os.path.exists(ds_info_path):
        with open(ds_info_path, "r", encoding="utf-8") as f:
            ds_infos = json.load(f)
    else:
        ds_infos = {}

    ds_infos[sft_data_name] = {
        "file_name": data_file,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "videos": "videos"
        }
    }

    with open(ds_info_path, "w", encoding="utf-8") as f:
        json.dump(ds_infos, f, indent=2, ensure_ascii=False)
    print(f"[ok] dataset_info.json updated → {ds_info_path}")


def main(sft_data_name: str):
    data_file  = f"{sft_data_name}.json"
    video_zip  = f"{sft_data_name}.zip"

    data_url = f"https://huggingface.co/datasets/{HF_DATASET_USER}/{HF_DATASET_NAME}/resolve/main/{data_file}"
    zip_url = f"https://huggingface.co/datasets/{HF_DATASET_USER}/{HF_VIDEO_NAME}/resolve/main/{video_zip}"

    data_save = os.path.join("data", data_file)
    zip_save = os.path.join("data", video_zip)
    video_dir = os.path.join("data", "videos")

    download_file(data_url, data_save)
    with open(data_save, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = [x for x in data if '000241_a.mp4' not in x['videos'][0]]
    with open(data_save, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    download_file(zip_url, zip_save)

    update_dataset_info(sft_data_name, data_file)

    os.makedirs(video_dir, exist_ok=True)
    try:
        with ZipFile(zip_save) as zf:
            zf.extractall(video_dir)
        print(f"[ok] Unzipped → {video_dir}")
    except BadZipFile as e:
        print(f"[error] Bad zip file: {e}")
    os.remove(zip_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT JSON & ZIP")
    parser.add_argument("--sft_data_name", required=True)
    args = parser.parse_args()
    main(args.sft_data_name)
