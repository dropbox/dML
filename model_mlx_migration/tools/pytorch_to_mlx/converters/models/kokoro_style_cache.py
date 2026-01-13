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

"""
Kokoro Style Parameter Cache (N2 Optimization)

Caches precomputed AdaIN fc(style) outputs for a given voice embedding.
Since the style vector is constant for a given voice, the fc() linear
transformations in all AdaIN layers produce identical outputs on every call.

This optimization precomputes all fc(style) outputs ONCE and reuses them,
eliminating redundant matrix multiplications.

Expected speedup: 5-8% (depending on sequence length)
Status: LOSSLESS - mathematically identical output

Usage:
    model = KokoroModel(config)
    model.load_weights(...)

    # One-time per voice:
    voice = load_voice_embedding("af")
    cache = precompute_style_cache(model, voice)

    # At inference - use cached style params:
    audio = model_with_cache(model, input_ids, voice, cache)
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class StyleCache:
    """
    Cached style parameters for all AdaIN layers.

    Structure:
        cache[layer_path] = (gamma, beta)  # Pre-split outputs of fc(style)

    Example layer_paths:
        - "predictor.text_encoder.lstms_1.fc"
        - "predictor.F0_0.norm1.fc"
        - "decoder.encode.norm1.fc"
        - "decoder.generator.resblocks_0.adain1_0.fc"
    """
    # Voice embedding that this cache was computed for
    voice_embedding: mx.array

    # Cached fc outputs: layer_path -> (gamma, beta) both [batch, channels]
    fc_outputs: dict[str, tuple[mx.array, mx.array]] = field(default_factory=dict)

    # Statistics
    num_layers_cached: int = 0
    total_params_saved: int = 0

    def get(self, layer_path: str) -> tuple[mx.array, mx.array] | None:
        """Get cached (gamma, beta) for a layer, or None if not cached."""
        return self.fc_outputs.get(layer_path)

    def put(self, layer_path: str, gamma: mx.array, beta: mx.array) -> None:
        """Cache (gamma, beta) for a layer."""
        self.fc_outputs[layer_path] = (gamma, beta)
        self.num_layers_cached += 1
        self.total_params_saved += gamma.size + beta.size


def _collect_adain_layers(
    module: nn.Module,
    prefix: str = "",
) -> dict[str, nn.Module]:
    """
    Recursively collect all AdaIN-like layers with fc attributes.

    Returns dict mapping layer_path -> module for layers that have:
    - fc attribute (AdaIN, AdaLayerNorm)
    - fc_style attribute (ProsodyConditionedAdaIN)
    """
    layers = {}

    # Check if this module has fc (AdaIN, AdaLayerNorm, AdaINStyleLinear)
    if hasattr(module, 'fc') and isinstance(module.fc, nn.Linear):
        layers[prefix] = module

    # Check for fc_style (ProsodyConditionedAdaIN)
    if hasattr(module, 'fc_style') and isinstance(module.fc_style, nn.Linear):
        layers[f"{prefix}_style"] = module

    # Recurse into children
    for name, child in module.named_modules():
        if name and child is not module:
            child_prefix = f"{prefix}.{name}" if prefix else name
            # Avoid double-counting - only process direct attributes
            if hasattr(module, name.split('.')[0]):
                child_layers = _collect_adain_layers(child, child_prefix)
                layers.update(child_layers)

    return layers


def _get_modules_with_fc(model: nn.Module) -> dict[str, tuple[nn.Module, str]]:
    """
    Find all modules with fc or fc_style linear layers.

    Returns: dict mapping path -> (module, fc_attr_name)
    """
    result = {}

    def traverse(mod: nn.Module, path: str):
        # Check for fc attribute
        if hasattr(mod, 'fc') and isinstance(getattr(mod, 'fc', None), nn.Linear):
            result[path] = (mod, 'fc')

        # Check for fc_style attribute
        if hasattr(mod, 'fc_style') and isinstance(getattr(mod, 'fc_style', None), nn.Linear):
            result[f"{path}.fc_style"] = (mod, 'fc_style')

        # Recurse into named children
        for name, child in mod.named_modules():
            if name:  # Skip self
                child_path = f"{path}.{name}" if path else name
                # MLX modules - check if it's a Module instance
                if isinstance(child, nn.Module) and child is not mod:
                    traverse(child, child_path)

    traverse(model, "")
    return result


def precompute_style_cache(
    model: nn.Module,
    voice: mx.array,
    verbose: bool = False,
) -> StyleCache:
    """
    Precompute and cache all AdaIN fc(style) outputs for a voice embedding.

    This traverses the model, finds all AdaIN-like layers, and computes
    their fc(style) outputs. The cache can then be used at inference to
    avoid redundant matrix multiplications.

    Args:
        model: KokoroModel instance
        voice: Voice embedding [batch, 256] - will be split into style/speaker
        verbose: Print cache statistics

    Returns:
        StyleCache with precomputed (gamma, beta) for all AdaIN layers
    """
    cache = StyleCache(voice_embedding=voice)

    # Voice embedding split: first 128 dims for decoder (style), rest for predictor (speaker)
    style = voice[:, :128]  # [batch, 128]
    speaker = voice[:, 128:]  # [batch, 128]

    # Predictor components use 'speaker' (128-dim)
    # Decoder/Generator use 'style' (128-dim from voice[:, :128])

    # Manually cache key AdaIN layers by traversing known architecture
    # This is more reliable than dynamic traversal for MLX modules

    cached_paths = []

    # === PREDICTOR: Uses speaker (voice[:, 128:]) ===
    if hasattr(model, 'predictor'):
        predictor = model.predictor

        # text_encoder.lstms_1, lstms_3, lstms_5 are PredictorTextEncoderFC
        # which have AdaLayerNorm with fc
        for name in ['lstms_1', 'lstms_3', 'lstms_5']:
            if hasattr(predictor.text_encoder, name):
                layer = getattr(predictor.text_encoder, name)
                if hasattr(layer, 'fc'):
                    h = layer.fc(speaker)  # [batch, hidden_dim * 2]
                    gamma, beta = mx.split(h, 2, axis=-1)
                    path = f"predictor.text_encoder.{name}"
                    cache.put(path, gamma, beta)
                    cached_paths.append(path)

        # F0_0, F0_1, F0_2 are AdainResBlk1d with norm1, norm2
        # NOTE: F0 predictor uses SPEAKER (voice[:, 128:]), not style!
        for name in ['F0_0', 'F0_1', 'F0_2']:
            if hasattr(predictor, name):
                block = getattr(predictor, name)
                for norm_name in ['norm1', 'norm2']:
                    if hasattr(block, norm_name):
                        norm = getattr(block, norm_name)
                        if hasattr(norm, 'fc'):
                            h = norm.fc(speaker)  # F0 predictor uses speaker
                            gamma, beta = mx.split(h, 2, axis=-1)
                            path = f"predictor.{name}.{norm_name}"
                            cache.put(path, gamma, beta)
                            cached_paths.append(path)

        # N_0, N_1, N_2 are AdainResBlk1d with norm1, norm2
        # NOTE: N predictor uses SPEAKER (voice[:, 128:]), not style!
        for name in ['N_0', 'N_1', 'N_2']:
            if hasattr(predictor, name):
                block = getattr(predictor, name)
                for norm_name in ['norm1', 'norm2']:
                    if hasattr(block, norm_name):
                        norm = getattr(block, norm_name)
                        if hasattr(norm, 'fc'):
                            h = norm.fc(speaker)  # N predictor uses speaker
                            gamma, beta = mx.split(h, 2, axis=-1)
                            path = f"predictor.{name}.{norm_name}"
                            cache.put(path, gamma, beta)
                            cached_paths.append(path)

    # === DECODER: Uses style (voice[:, :128]) ===
    if hasattr(model, 'decoder'):
        decoder = model.decoder

        # encode is AdainResBlk1d
        if hasattr(decoder, 'encode'):
            for norm_name in ['norm1', 'norm2']:
                if hasattr(decoder.encode, norm_name):
                    norm = getattr(decoder.encode, norm_name)
                    if hasattr(norm, 'fc'):
                        h = norm.fc(style)
                        gamma, beta = mx.split(h, 2, axis=-1)
                        path = f"decoder.encode.{norm_name}"
                        cache.put(path, gamma, beta)
                        cached_paths.append(path)

        # decode_0, decode_1, decode_2, decode_3 are AdainResBlk1d
        for name in ['decode_0', 'decode_1', 'decode_2', 'decode_3']:
            if hasattr(decoder, name):
                block = getattr(decoder, name)
                for norm_name in ['norm1', 'norm2']:
                    if hasattr(block, norm_name):
                        norm = getattr(block, norm_name)
                        if hasattr(norm, 'fc'):
                            h = norm.fc(style)
                            gamma, beta = mx.split(h, 2, axis=-1)
                            path = f"decoder.{name}.{norm_name}"
                            cache.put(path, gamma, beta)
                            cached_paths.append(path)

        # Generator: noise_res_i are AdaINResBlock1dStyled
        # resblocks_i are AdaINResBlock1dStyled
        if hasattr(decoder, 'generator'):
            gen = decoder.generator

            # noise_res blocks (typically 2)
            for i in range(2):
                name = f"noise_res_{i}"
                if hasattr(gen, name):
                    block = getattr(gen, name)
                    # AdaINResBlock1dStyled has adain1_0,1,2 and adain2_0,1,2
                    for j in range(3):
                        for adain_name in [f'adain1_{j}', f'adain2_{j}']:
                            if hasattr(block, adain_name):
                                adain = getattr(block, adain_name)
                                if hasattr(adain, 'fc'):
                                    h = adain.fc(style)
                                    # AdaINStyleLinear output is channels*2
                                    channels = adain.channels
                                    gamma = h[:, :channels]
                                    beta = h[:, channels:]
                                    path = f"decoder.generator.{name}.{adain_name}"
                                    cache.put(path, gamma, beta)
                                    cached_paths.append(path)

            # resblocks (typically 6)
            num_resblocks = getattr(gen, '_num_resblocks', 6)
            for i in range(num_resblocks):
                name = f"resblocks_{i}"
                if hasattr(gen, name):
                    block = getattr(gen, name)
                    for j in range(3):
                        for adain_name in [f'adain1_{j}', f'adain2_{j}']:
                            if hasattr(block, adain_name):
                                adain = getattr(block, adain_name)
                                if hasattr(adain, 'fc'):
                                    h = adain.fc(style)
                                    channels = adain.channels
                                    gamma = h[:, :channels]
                                    beta = h[:, channels:]
                                    path = f"decoder.generator.{name}.{adain_name}"
                                    cache.put(path, gamma, beta)
                                    cached_paths.append(path)

    # Evaluate all cached values
    mx.eval(cache.voice_embedding)
    for gamma, beta in cache.fc_outputs.values():
        mx.eval(gamma, beta)

    if verbose:
        print("Style cache statistics:")
        print(f"  Layers cached: {cache.num_layers_cached}")
        print(f"  Total params saved: {cache.total_params_saved:,}")
        print(f"  Memory saved per inference: ~{cache.total_params_saved * 4 / 1024:.1f} KB")
        print(f"  Cached paths: {len(cached_paths)}")

    return cache


def verify_style_match(cache: StyleCache, voice: mx.array, rtol: float = 1e-5) -> bool:
    """
    Verify that voice embedding matches the cached embedding.

    Args:
        cache: StyleCache instance
        voice: Voice embedding to check
        rtol: Relative tolerance for comparison

    Returns:
        True if voice matches cached embedding
    """
    if cache.voice_embedding is None:
        return False

    if cache.voice_embedding.shape != voice.shape:
        return False

    diff = mx.abs(cache.voice_embedding - voice)
    max_diff = mx.max(diff)

    return float(max_diff) < rtol


# ============================================================================
# Cache-enabled inference functions
# ============================================================================

def adain_with_cache(
    norm: nn.Module,
    x: mx.array,
    s: mx.array,
    cache: StyleCache | None,
    cache_path: str,
) -> mx.array:
    """
    Apply AdaIN with optional style cache.

    If cache is provided and contains the path, uses cached gamma/beta.
    Otherwise computes them from scratch.

    Args:
        norm: AdaIN or AdaLayerNorm module
        x: Input tensor [batch, length, channels]
        s: Style vector [batch, style_dim]
        cache: Optional StyleCache
        cache_path: Path to look up in cache

    Returns:
        Normalized and styled output
    """
    # Try to get from cache
    if cache is not None:
        cached = cache.get(cache_path)
        if cached is not None:
            gamma, beta = cached
            gamma = gamma[:, None, :]  # [batch, 1, channels]
            beta = beta[:, None, :]

            # Instance normalization (same as original)
            mean = mx.mean(x, axis=1, keepdims=True)
            var = mx.var(x, axis=1, keepdims=True)
            x_norm = (x - mean) / mx.sqrt(var + norm.eps)

            # Apply InstanceNorm affine if present
            if hasattr(norm, 'norm_weight'):
                x_norm = norm.norm_weight * x_norm + norm.norm_bias

            # Apply cached adaptive scale and shift
            return (1 + gamma) * x_norm + beta

    # Fall back to normal computation
    return norm(x, s)


def estimate_cache_savings(model: nn.Module) -> dict[str, int]:
    """
    Estimate computational savings from style caching.

    Returns:
        Dict with:
        - num_fc_layers: Number of fc layers that can be cached
        - total_fc_ops: Total multiply-add operations saved per inference
        - memory_bytes: Memory required for cache
    """
    num_layers = 0
    total_ops = 0
    total_params = 0

    # This is a rough estimate based on typical Kokoro architecture
    # Predictor: 3 AdaLayerNorm + 6*2 AdaIN (F0/N blocks)
    # Decoder: 1*2 + 4*2 AdaIN (encode + decode blocks)
    # Generator: 2*3*2 + 6*3*2 AdaIN (noise_res + resblocks)

    predictor_adaln = 3  # lstms_1, lstms_3, lstms_5
    predictor_adain = 12  # 6 blocks * 2 norms
    decoder_adain = 10  # encode + 4 decode blocks
    generator_adain = 48  # 2 noise_res * 6 + 6 resblocks * 6

    num_layers = predictor_adaln + predictor_adain + decoder_adain + generator_adain

    # Each fc is Linear(128, channels*2)
    # Typical channels: 512, 256 for predictor/decoder, 128-512 for generator
    avg_channels = 400
    style_dim = 128

    # Operations per fc: style_dim * (2 * channels) multiply-adds
    ops_per_fc = style_dim * (2 * avg_channels)
    total_ops = num_layers * ops_per_fc

    # Memory: 2 * channels floats per layer (gamma + beta)
    total_params = num_layers * 2 * avg_channels

    return {
        'num_fc_layers': num_layers,
        'total_fc_ops': total_ops,
        'memory_bytes': total_params * 4,  # float32
    }
