#!/usr/bin/env python3
"""
Pre-download all datasets to the Scratch directory.
Run this once before submitting training jobs.

Usage:
    python download_datasets.py
"""

import os
from datasets import load_dataset

HF_CACHE_DIR = "/home/zceccgr/Scratch/huggingface_cache"

def download_all_datasets():
    # Ensure cache directory exists
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    print(f"Downloading datasets to: {HF_CACHE_DIR}\n")

    # 1. Download nq_open (for pre-training)
    print("=" * 50)
    print("Downloading nq_open dataset...")
    print("=" * 50)
    nq_train = load_dataset("nq_open", split="train", cache_dir=HF_CACHE_DIR)
    nq_val = load_dataset("nq_open", split="validation", cache_dir=HF_CACHE_DIR)
    print(f"nq_open train: {len(nq_train)} examples")
    print(f"nq_open validation: {len(nq_val)} examples")
    print("Done!\n")

    # 2. Download WikiText-103 (for pre-training)
    print("=" * 50)
    print("Downloading WikiText-103 dataset...")
    print("=" * 50)
    wiki_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=HF_CACHE_DIR)
    wiki_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", cache_dir=HF_CACHE_DIR)
    print(f"WikiText-103 train: {len(wiki_train)} examples")
    print(f"WikiText-103 validation: {len(wiki_val)} examples")
    print("Done!\n")

    # 3. Download SQuAD (for fine-tuning)
    print("=" * 50)
    print("Downloading SQuAD dataset...")
    print("=" * 50)
    squad_train = load_dataset("squad", split="train", cache_dir=HF_CACHE_DIR)
    squad_val = load_dataset("squad", split="validation", cache_dir=HF_CACHE_DIR)
    print(f"SQuAD train: {len(squad_train)} examples")
    print(f"SQuAD validation: {len(squad_val)} examples")
    print("Done!\n")

    print("=" * 50)
    print("All datasets downloaded successfully!")
    print(f"Cache location: {HF_CACHE_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    download_all_datasets()