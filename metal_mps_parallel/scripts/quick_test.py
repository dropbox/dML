#!/usr/bin/env python3
"""Quick crash test."""
import torch
import torch.nn as nn
import threading
import sys

model = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    batch_first=True, dropout=0
).to('mps').eval()

x = torch.randn(1, 4, 64, device='mps')

def worker(iterations=10):
    for i in range(iterations):
        with torch.no_grad():
            y = model(x)
        torch.mps.synchronize()

threads = [threading.Thread(target=worker) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("PASS")
