#!/usr/bin/env python3
"""Generate reference data for RelPositionalEncoding comparison.

Uses RelPositionalEncoding (from zipformer.py) which SETS the last column to 1.0.
This is what the actual Zipformer2Encoder uses, not CompactRelPositionalEncoding.
"""

import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration/src')

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

from models.zipformer.zipformer import RelPositionalEncoding


def main():
    # Match the test parameters from C++ test
    seq_len = 327  # Same as embed output after Conv2dSubsampling
    batch_size = 1
    pos_dim = 48   # Default for streaming model

    # Create the positional encoding module
    # Use RelPositionalEncoding which the Zipformer2Encoder actually uses
    pos_enc = RelPositionalEncoding(pos_dim=pos_dim)

    # Create dummy input with correct shape (seq_len, batch_size, embed_dim)
    x = mx.zeros((seq_len, batch_size, 192))  # d_model for stage 0

    # Compute positional encoding
    # RelPositionalEncoding.__call__ takes just x, returns (batch, 2*seq-1, pos_dim)
    pos_emb = pos_enc(x)
    mx.eval(pos_emb)

    print(f"Input shape (seq_len, batch, d_model): {x.shape}")
    print(f"Output pos_emb shape: {pos_emb.shape}")  # Should be (1, 2*seq_len-1, pos_dim)

    pos_emb_np = np.array(pos_emb)

    # Check statistics
    print("\nPositional encoding statistics:")
    print(f"  Shape: {pos_emb_np.shape}")
    print(f"  Min: {pos_emb_np.min():.6f}")
    print(f"  Max: {pos_emb_np.max():.6f}")
    print(f"  Mean: {pos_emb_np.mean():.6f}")

    # Check last column (should have bias of +1.0)
    last_col = pos_emb_np[0, :, -1]
    print("\nLast column (with bias):")
    print(f"  Min: {last_col.min():.6f}")
    print(f"  Max: {last_col.max():.6f}")
    print(f"  Mean: {last_col.mean():.6f}")
    print(f"  First 5 values: {last_col[:5]}")
    print(f"  Center values: {last_col[seq_len-3:seq_len+2]}")  # Around position 0

    # Check center position (position 0 in relative encoding)
    center_idx = seq_len - 1  # Position for relative offset = 0
    center_vec = pos_emb_np[0, center_idx, :]
    print("\nCenter position (rel offset=0):")
    print(f"  First 8 values: {center_vec[:8]}")
    print(f"  Last value (with bias): {center_vec[-1]:.6f}")

    # Save reference
    output_path = "/Users/ayates/model_mlx_migration/checkpoints/zipformer/en-streaming/pos_enc_reference.safetensors"
    save_file({
        "pos_emb": pos_emb_np.astype(np.float32),
        "seq_len": np.array([seq_len], dtype=np.int32),
        "batch_size": np.array([batch_size], dtype=np.int32),
        "pos_dim": np.array([pos_dim], dtype=np.int32),
    }, output_path)

    print(f"\nSaved reference to: {output_path}")


if __name__ == "__main__":
    main()
