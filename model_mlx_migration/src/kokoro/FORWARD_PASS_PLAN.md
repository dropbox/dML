# Kokoro C++ Forward Pass Implementation Plan

This document outlines the implementation plan for the MLX C++ forward pass.

## Weight Key Structure

Weights are loaded from `weights.safetensors` with the following prefixes:

### BERT/ALBERT Weights
```
bert.embeddings.word_embeddings.weight          [178, 128]
bert.embeddings.position_embeddings.weight      [512, 128]
bert.embeddings.token_type_embeddings.weight    [2, 128]
bert.embeddings.LayerNorm.weight                [128]
bert.embeddings.LayerNorm.bias                  [128]
bert.embeddings.embedding_hidden_mapping.weight [768, 128]
bert.embeddings.embedding_hidden_mapping.bias   [768]

bert.encoder.albert_layer.attention.query.weight    [768, 768]
bert.encoder.albert_layer.attention.query.bias      [768]
bert.encoder.albert_layer.attention.key.weight      [768, 768]
bert.encoder.albert_layer.attention.key.bias        [768]
bert.encoder.albert_layer.attention.value.weight    [768, 768]
bert.encoder.albert_layer.attention.value.bias      [768]
bert.encoder.albert_layer.attention.dense.weight    [768, 768]
bert.encoder.albert_layer.attention.dense.bias      [768]
bert.encoder.albert_layer.attention.LayerNorm.weight [768]
bert.encoder.albert_layer.attention.LayerNorm.bias   [768]

bert.encoder.albert_layer.ffn.weight            [2048, 768]
bert.encoder.albert_layer.ffn.bias              [2048]
bert.encoder.albert_layer.ffn_output.weight     [768, 2048]
bert.encoder.albert_layer.ffn_output.bias       [768]
bert.encoder.albert_layer.full_layer_layer_norm.weight [768]
bert.encoder.albert_layer.full_layer_layer_norm.bias   [768]

bert.pooler.weight                              [768, 768]
bert.pooler.bias                                [768]
```

### Decoder Weights (decoder.*)
Large number of convolution and residual block weights for ISTFTNet.

## MLX C++ API Mapping

### Embedding
```cpp
// Python: nn.Embedding(n_token, dim)
// C++: Manual gather from weight matrix
mx::array embed(const mx::array& tokens, const mx::array& weight) {
    return mx::take(weight, tokens, 0);  // [batch, seq] -> [batch, seq, dim]
}
```

### Linear Layer
```cpp
// Python: nn.Linear(in_features, out_features)
// C++: matmul + add
mx::array linear(const mx::array& x, const mx::array& weight, const mx::array& bias) {
    auto out = mx::matmul(x, mx::transpose(weight));  // [B, T, in] @ [out, in].T
    return out + bias;
}
```

### LayerNorm
```cpp
// Python: nn.LayerNorm(dim)
// C++: normalize + scale + shift
mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias, float eps = 1e-5) {
    auto mean = mx::mean(x, -1, true);
    auto var = mx::var(x, -1, true);
    auto normalized = (x - mean) / mx::sqrt(var + eps);
    return normalized * weight + bias;
}
```

### Attention
```cpp
// Q, K, V projections
auto q = linear(x, weights.get("bert.encoder.albert_layer.attention.query.weight"),
                   weights.get("bert.encoder.albert_layer.attention.query.bias"));
auto k = linear(x, weights.get("bert.encoder.albert_layer.attention.key.weight"),
                   weights.get("bert.encoder.albert_layer.attention.key.bias"));
auto v = linear(x, weights.get("bert.encoder.albert_layer.attention.value.weight"),
                   weights.get("bert.encoder.albert_layer.attention.value.bias"));

// Reshape for multi-head attention
// [B, T, H*D] -> [B, num_heads, T, head_dim]
q = mx::transpose(mx::reshape(q, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
k = mx::transpose(mx::reshape(k, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
v = mx::transpose(mx::reshape(v, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});

// Attention scores
float scale = std::sqrt((float)head_dim);
auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) / scale;
auto attn_weights = mx::softmax(scores, -1);

// Apply attention
auto context = mx::matmul(attn_weights, v);  // [B, num_heads, T, head_dim]
context = mx::transpose(context, {0, 2, 1, 3});  // [B, T, num_heads, head_dim]
context = mx::reshape(context, {batch, seq_len, hidden_size});  // [B, T, H]
```

## Implementation Order

### Phase 1: BERT Embeddings
1. word_embeddings (embedding lookup)
2. position_embeddings (position ids)
3. token_type_embeddings (zeros for single segment)
4. LayerNorm
5. embedding_hidden_mapping (project 128 -> 768)

### Phase 2: BERT Encoder
1. AlbertAttention (Q/K/V, multi-head, output projection)
2. FFN (2-layer MLP with GELU)
3. Repeat for num_layers (12)

### Phase 3: Predictor
1. Text encoder (Conv1d + BiLSTM)
2. Duration predictor
3. F0/Noise predictor

### Phase 4: Decoder/Generator
1. ISTFTNet vocoder
2. Source module (harmonic generation)
3. Audio output

## Testing Strategy

1. Export intermediate tensors from Python
2. Implement C++ forward pass stage by stage
3. Compare intermediate outputs with tolerance
4. Use Whisper transcription for final validation

## Current Status (as of #381)

**COMPLETE** - C++ pipeline produces correct speech. Whisper transcription tests pass.

### Verification (2025-12-14)
- "Hello world!" → Whisper: "Hello world" ✓
- "Thank you for your help" → Whisper: "Thank you for your help." ✓
- "The quick brown fox jumps over the lazy dog" → Whisper: exact match ✓
- Performance: 25-32x real-time (58-113ms for typical phrases)
- Max amplitude: ~0.3-0.4 (properly normalized)

### Component Status
- Infrastructure: DONE (model loading, G2P, tokenizer)
- BERT Embeddings: DONE (word + position + type + LayerNorm)
- BERT Encoder: DONE (12 ALBERT layers with shared weights)
- Text Encoder: DONE (Conv1d + BiLSTM)
- Predictor: DONE (duration, alignment, F0/N prediction)
- Decoder: DONE (full ISTFTNet vocoder)
- m_source: DONE (harmonic generation from F0)
- ISTFT: DONE (correct audio output)
- Shape bucketing: DEFERRED (low priority optimization - requires cmake to rebuild C++)

### Key Fixes (#377-380)
1. **#377-378**: F0/N predictor operation order (AdaIN→ReLU→Conv)
2. **#379**: LeakyReLU slope 0.2→0.01 before conv_post
3. **#380**: GELU activation (gelu_new tanh approximation), learned upsampling (ConvTranspose1d)

Changes in #377-378:
1. Fixed F0_0, F0_2, N_0, N_2 operation order: AdaIN->ReLU->Conv (was Conv->AdaIN->ReLU)
2. Added rsqrt(2) scaling for all predictor residual blocks
3. Fixed style/speaker usage: predictor uses speaker, decoder uses style

Key metrics (after #378):
- Duration: 1.55s (correct)
- Max amplitude: 141.19 (clipped to 1.0 in WAV)
- F0 range: -9.18 to 179.71 Hz (82 voiced frames out of 124)
- Whisper transcription: "FUCK memories" (improved from empty)

Changes in #377-378 (F0/N predictor fixes):
1. Fixed F0_0, F0_2, N_0, N_2 operation order: AdaIN->ReLU->Conv
2. Added rsqrt(2) scaling for residual blocks
3. Fixed style/speaker: text_encoder and F0/N use speaker, decoder uses style

Changes in #376:
1. **VERIFIED**: All decode blocks (encode, decode_0-3) match Python exactly (max_diff < 0.0001)
2. **VERIFIED**: Generator ups_0 matches Python exactly
3. **VERIFIED**: Generator resblock 3 (rb3) matches Python exactly
   - The "explosion" to max=136 is expected behavior - Python produces same values
   - This is NOT a bug in C++ - it's the model's actual behavior
4. Added debug scripts for step-by-step comparison (debug_decoder_blocks.py, debug_generator_rb3.py)
5. Added save_npy() helper for exporting tensors for Python comparison
6. Fixed Python debug script operation order (was adain->conv->relu, should be adain->relu->conv)

Changes in #375:
1. **MAJOR BUG FIX**: Decoder was receiving wrong style vector!
   - Was using speaker[0:128] (voice_embed[128:256]) instead of style[0:128] (voice_embed[0:128])
   - Fixed by passing correct style directly to decoder_forward
   - This caused gamma values to be [-4.7, 3.3] instead of [-1.6, 0.4], causing massive amplification
2. Added debug output for style vector verification
3. Added adain debug output (gamma, beta, x_norm)

Remaining issues to debug:
1. **Audio amplitude explosion** - Max amplitude 141x expected range
   - Generator resblocks produce large values (expected - Python does too)
   - Need to investigate final normalization
   - May need different clipping/limiting approach
2. **Harmonic source (m_source)** - Could contribute to quality
   - F0 upsampling method
   - Phase accumulation
   - Harmonic weighting with l_linear
3. **F0 still has negative values** - Should be >= 0 Hz
   - Currently -9 to 180 Hz
   - May need clamping at predictor output

## Performance Measurements

- Model load: ~35ms
- Full pipeline synthesis: ~55-65ms for "Hello world" (~25x real-time)
- Output samples: 37200 for "Hello world" (1.55s audio)
- Whisper transcription: "FUCK memories" (partial recognition)

## Decoder Structure (for next AI)

### Overview
The decoder has 310+ weights. Currently producing test tone output.

### Weight Groups

**decoder.asr_res**: Conv1d [64, 512, 1] - Processes ASR features (DONE)

**decoder.encode**: Initial encoding block
- conv1, conv1x1, conv2: [1024, 514, 3], [1024, 514, 1], [1024, 1024, 3]
- norm1, norm2: AdaLayerNorm FC layers

**decoder.decode_{0-3}**: Decode blocks with optional upsampling
- Each has conv1, conv1x1, conv2, norm1, norm2
- decode_3 has pool (average pooling)

**decoder.generator**: ISTFTNet vocoder (~210 weights)
- m_source: l_linear [1, 9] - harmonic source generation
- noise_convs_{0,1}: Process source signal
- noise_res_{0,1}: AdaIN residual blocks for noise
- resblocks_{0-5}: Main processing blocks (6 total, each with 3 dilations)
- ups_{0,1}: Transpose conv upsampling
  - ups_0: [512, 256, 20] upsample 10x
  - ups_1: [256, 128, 12] upsample 6x
- conv_post: [22, 128, 7] final output to spectrogram

### Generator Flow (Python reference: kokoro.py:980+)

1. total_upp = 10 * 6 * 5 = 300 (upsample_rates * istft_hop)
2. har_source, noise, uv = m_source(f0, total_upp)
3. source = STFT(har_source) -> 22 channels
4. For each upsample stage (2 total):
   - x = leaky_relu(x)
   - x_source = noise_conv(source)
   - x_source = noise_res(x_source, style)
   - x = transpose_conv(x)
   - if last stage: reflect pad
   - x = x + x_source
   - x = sum(resblocks[stage*3:(stage+1)*3](x, style)) / 3
5. x = leaky_relu(x) # 0.01 slope
6. x = conv_post(x) -> [batch, frames, 22]
7. mag = exp(x[:,:11])
8. phase = sin(x[:,11:])
9. audio = ISTFT(mag * exp(i*phase))

### Implementation Priority

1. Generator ups_0, ups_1 (transpose conv)
2. conv_post (final spectrogram)
3. Simple ISTFT (ignore source initially)
4. Add resblocks incrementally
5. Add noise_convs/noise_res
6. Add m_source for F0-based harmonic generation

## Correct Weight Keys (lowercase!)

Note: Weight keys use lowercase (e.g., `layer_norm` not `LayerNorm`):
```
bert.embeddings.layer_norm.weight       # NOT LayerNorm
bert.encoder.albert_layer.attention.layer_norm.weight
bert.encoder.embedding_hidden_mapping_in.weight  # 128->768 projection
```

## Notes

- MLX C++ uses row-major layout (same as Python)
- Weights are pre-folded (WeightNorm computed offline)
- ALBERT shares weights across layers (single layer repeated)
- Dropout disabled at inference (pass deterministic=true)
