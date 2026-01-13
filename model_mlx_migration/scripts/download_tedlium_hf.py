#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download TED-LIUM 3 from HuggingFace."""

import os
from huggingface_hub import hf_hub_download

OUTPUT_DIR = "/Users/ayates/model_mlx_migration/data/emotion_punctuation/TEDLIUM_release3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "TEDLIUM_release3/legacy/dev.tar.gz",
    "TEDLIUM_release3/legacy/test.tar.gz",
    "TEDLIUM_release3/legacy/train_1.tar.gz",
    "TEDLIUM_release3/legacy/train_2.tar.gz",
]

for f in FILES:
    filename = os.path.basename(f)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        print(f"[SKIP] {filename} already exists")
        continue

    print(f"[DOWNLOAD] {filename}...")
    try:
        path = hf_hub_download(
            repo_id="LIUM/tedlium",
            filename=f,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[DONE] {filename} -> {path}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

print("\nTED-LIUM download complete!")
