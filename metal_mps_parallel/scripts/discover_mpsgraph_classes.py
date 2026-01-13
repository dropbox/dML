#!/usr/bin/env python3
"""Discover MPSGraph-related classes used by PyTorch."""

import torch
import torch.nn as nn
import ctypes
from ctypes import CDLL, c_void_p, c_char_p, c_int

# Load ObjC runtime
libobjc = CDLL('/usr/lib/libobjc.A.dylib')
libobjc.objc_getClassList.argtypes = [c_void_p, c_int]
libobjc.objc_getClassList.restype = c_int
libobjc.class_getName.argtypes = [c_void_p]
libobjc.class_getName.restype = c_char_p

def get_all_classes():
    """Get all loaded ObjC classes."""
    count = libobjc.objc_getClassList(None, 0)
    classes = (c_void_p * count)()
    libobjc.objc_getClassList(classes, count)

    names = []
    for cls in classes:
        if cls:
            try:
                name = libobjc.class_getName(cls)
                if name:
                    names.append(name.decode('utf-8'))
            except:
                pass
    return names

print("Loading PyTorch and triggering MPS/MPSGraph usage...")

# Force MPSGraph usage by running TransformerEncoderLayer
device = torch.device("mps")
model = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    batch_first=True, dropout=0
).to(device).eval()

x = torch.randn(1, 4, 64, device=device)
with torch.no_grad():
    y = model(x)
torch.mps.synchronize()

print("\n=== MPSGraph-related classes ===")
classes = get_all_classes()
mpsgraph_classes = sorted([c for c in classes if 'MPSGraph' in c])
for c in mpsgraph_classes:
    print(f"  {c}")

print("\n=== AGX-related classes (first 30) ===")
agx_classes = sorted([c for c in classes if 'AGX' in c])
for c in agx_classes[:30]:
    print(f"  {c}")
