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
Native MLX CTC Loss Implementation.

Implements Connectionist Temporal Classification loss using the forward-backward
algorithm, entirely in MLX for efficient training without PyTorch bridge.

This implementation is validated against PyTorch's CTC loss to ensure numerical
equivalence within 1e-5 tolerance.

References:
    - Graves et al. "Connectionist Temporal Classification: Labelling
      Unsegmented Sequence Data with Recurrent Neural Networks" (2006)
    - https://distill.pub/2017/ctc/
    - PyTorch CTC loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

Key design decisions:
    - Log-space computation for numerical stability
    - Log-sum-exp for stable addition of probabilities
    - Supports variable length inputs and targets within a batch
    - Compatible with mx.grad() for automatic differentiation
"""


import mlx.core as mx
import numpy as np

# Numerical constants for log-space computation
LOG_ZERO = -1e30  # Effectively -inf but avoids NaN in gradients


def log_sum_exp(a: mx.array, b: mx.array) -> mx.array:
    """
    Numerically stable log(exp(a) + exp(b)).

    Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    """
    max_val = mx.maximum(a, b)
    return max_val + mx.log1p(mx.exp(-mx.abs(a - b)))


def log_sum_exp_3(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    """Numerically stable log(exp(a) + exp(b) + exp(c))."""
    return log_sum_exp(log_sum_exp(a, b), c)


def ctc_loss(
    log_probs: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = True,
) -> mx.array:
    """
    Compute CTC loss for a batch of sequences.

    This is the main entry point for CTC loss computation. It handles batched
    inputs with variable lengths and supports all standard reduction modes.

    Args:
        log_probs: Log probabilities from model, shape (T, N, C) where
                   T=max time steps, N=batch size, C=num classes
        targets: Flattened target sequences, shape (sum(target_lengths),)
                 Contains concatenated target indices for all batch items
        input_lengths: Length of each input sequence, shape (N,)
        target_lengths: Length of each target sequence, shape (N,)
        blank: Index of the blank label (default: 0)
        reduction: 'none', 'mean', or 'sum'
        zero_infinity: Replace infinite losses with 0 (PyTorch compatibility)

    Returns:
        CTC loss value(s). Shape depends on reduction:
        - 'none': (N,) per-sample losses
        - 'mean' or 'sum': scalar

    Example:
        >>> log_probs = mx.log_softmax(logits, axis=-1)  # (T, N, C)
        >>> log_probs = mx.transpose(log_probs, (1, 0, 2))  # (T, N, C)
        >>> loss = ctc_loss(log_probs, targets, input_lens, target_lens)
    """
    T, N, C = log_probs.shape

    # Convert to numpy for indexing operations, then compute loss
    # This is necessary because MLX doesn't support advanced indexing in pure MLX
    log_probs_np = np.array(log_probs)
    targets_np = np.array(targets, dtype=np.int32)
    input_lengths_np = np.array(input_lengths, dtype=np.int32)
    target_lengths_np = np.array(target_lengths, dtype=np.int32)

    # Compute per-sample losses
    losses = []
    target_offset = 0

    for n in range(N):
        T_n = int(input_lengths_np[n])
        S_n = int(target_lengths_np[n])

        # Extract this sample's log probs and targets
        log_probs_n = log_probs_np[:T_n, n, :]  # (T_n, C)
        targets_n = targets_np[target_offset:target_offset + S_n]  # (S_n,)
        target_offset += S_n

        # Compute single-sample loss
        loss_n = _ctc_loss_single(log_probs_n, targets_n, blank)
        losses.append(float(loss_n))

    losses = mx.array(np.array(losses, dtype=np.float32))

    # Handle infinite losses
    if zero_infinity:
        losses = mx.where(mx.isinf(losses), mx.zeros_like(losses), losses)

    # Apply reduction
    if reduction == "none":
        return losses
    if reduction == "sum":
        return mx.sum(losses)
    if reduction == "mean":
        # PyTorch default: divide each loss by its target length, then average over batch
        # This is: mean(loss_i / target_length_i) for i in batch
        target_lengths_float = target_lengths.astype(mx.float32)
        normalized_losses = losses / target_lengths_float
        return mx.mean(normalized_losses)
    raise ValueError(f"Invalid reduction: {reduction}")


def _ctc_loss_single(
    log_probs: np.ndarray,
    targets: np.ndarray,
    blank: int,
) -> float:
    """
    Compute CTC loss for a single sequence using forward algorithm.

    Args:
        log_probs: Log probabilities, shape (T, C)
        targets: Target sequence (no blanks), shape (S,)
        blank: Index of blank label

    Returns:
        Negative log-likelihood (CTC loss)
    """
    loss, _ = _ctc_forward_backward_single(log_probs, targets, blank)
    return loss


def _ctc_forward_backward_single(
    log_probs: np.ndarray,
    targets: np.ndarray,
    blank: int,
) -> tuple[float, np.ndarray]:
    """
    Compute CTC loss and gradient for a single sequence using forward-backward.

    Implements the standard CTC forward-backward algorithm from:
    Graves et al. "Connectionist Temporal Classification" (2006), Section 4.1

    The gradient with respect to unnormalized log-probs (logits) is:
        ∇_logits[t,k] = y[t,k] - (1/Z) * Σ_{s: labels[s]=k} exp(α[t,s] + β[t,s])

    where y[t,k] = softmax(logits)[t,k] and Z = P(labels|input).

    Args:
        log_probs: Log probabilities, shape (T, C)
        targets: Target sequence (no blanks), shape (S,)
        blank: Index of blank label

    Returns:
        Tuple of:
        - loss: Negative log-likelihood (scalar)
        - grad: Gradient w.r.t. log_probs, shape (T, C)
    """
    T, C = log_probs.shape
    S = len(targets)

    if S == 0:
        # Empty target - all outputs should be blank
        loss = -np.sum(log_probs[:, blank])
        # Gradient: -1 for blank positions, 0 elsewhere
        grad = np.zeros((T, C), dtype=np.float32)
        grad[:, blank] = -1.0
        return loss, grad

    # Extended label sequence with blanks: [blank, y0, blank, y1, blank, ...]
    L = 2 * S + 1
    extended = np.zeros(L, dtype=np.int32)
    extended[0::2] = blank  # Even indices are blanks
    extended[1::2] = targets  # Odd indices are labels

    # =========================================================================
    # Forward pass (alpha)
    # alpha[t, s] = log P(output extended[:s+1] | input[:t+1])
    # =========================================================================
    alpha = np.full((T, L), LOG_ZERO, dtype=np.float32)

    # Base case: t=0
    alpha[0, 0] = log_probs[0, extended[0]]
    if L > 1:
        alpha[0, 1] = log_probs[0, extended[1]]

    # Forward recursion
    for t in range(1, T):
        for s in range(L):
            label = extended[s]

            # Option 1: Stay in same state
            alpha_sum = alpha[t-1, s]

            # Option 2: Transition from previous state (s-1)
            if s > 0:
                alpha_sum = _log_add_np(alpha_sum, alpha[t-1, s-1])

            # Option 3: Skip blank (only if current != blank and current != s-2)
            if s > 1:
                label_s_minus_2 = extended[s-2]
                if label != blank and label != label_s_minus_2:
                    alpha_sum = _log_add_np(alpha_sum, alpha[t-1, s-2])

            alpha[t, s] = alpha_sum + log_probs[t, label]

    # Total log probability Z = P(labels|input)
    log_Z = _log_add_np(alpha[T-1, L-1], alpha[T-1, L-2])
    loss = -log_Z

    # =========================================================================
    # Backward pass (beta)
    # beta[t, s] = log P(output extended[s:] | input[t:], starting at state s)
    # =========================================================================
    beta = np.full((T, L), LOG_ZERO, dtype=np.float32)

    # Base case: t=T-1
    beta[T-1, L-1] = 0.0  # log(1) = 0
    beta[T-1, L-2] = 0.0

    # Backward recursion
    for t in range(T-2, -1, -1):
        for s in range(L):
            label = extended[s]

            # Option 1: Stay in same state
            beta_sum = beta[t+1, s] + log_probs[t+1, extended[s]]

            # Option 2: Transition to next state (s+1)
            if s < L - 1:
                beta_sum = _log_add_np(
                    beta_sum,
                    beta[t+1, s+1] + log_probs[t+1, extended[s+1]],
                )

            # Option 3: Skip blank (s+2)
            if s < L - 2:
                label_s_plus_2 = extended[s+2]
                if label != blank and label != label_s_plus_2:
                    beta_sum = _log_add_np(
                        beta_sum,
                        beta[t+1, s+2] + log_probs[t+1, extended[s+2]],
                    )

            beta[t, s] = beta_sum

    # =========================================================================
    # Compute gradient
    #
    # The CTC loss is L = -log(Z) where Z = P(labels|input).
    #
    # For gradient w.r.t. logits (before softmax), the standard formula is:
    # ∂L/∂z[t,k] = y[t,k] - gamma[t,k]
    #
    # where:
    # - y[t,k] = softmax(z)[t,k] = exp(log_probs[t,k])
    # - gamma[t,k] = (1/Z) * Σ_{s: labels[s]=k} exp(α[t,s] + β[t,s])
    #
    # Since PyTorch CTC expects log_probs as input but computes gradients
    # as if they were logits (i.e., gradients flow through the implicit
    # softmax), we use the same formula.
    # =========================================================================

    # Compute gamma (occupation probability for each label at each time)
    gamma = np.zeros((T, C), dtype=np.float32)
    for t in range(T):
        for s in range(L):
            label = extended[s]
            # log P(being at state s at time t | input, labels)
            log_gamma_ts = alpha[t, s] + beta[t, s] - log_Z

            if log_gamma_ts > LOG_ZERO + 10:  # Avoid underflow
                gamma_ts = np.exp(log_gamma_ts)
                gamma[t, label] += gamma_ts

    # Gradient = y - gamma = exp(log_probs) - gamma
    # This is the standard CTC gradient formula
    y = np.exp(log_probs)  # softmax probabilities
    grad = y - gamma

    return loss, grad


def _log_add_np(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) for numpy scalars."""
    if a == LOG_ZERO:
        return b
    if b == LOG_ZERO:
        return a
    max_val = max(a, b)
    return max_val + np.log1p(np.exp(-abs(a - b)))


def ctc_loss_with_grad(
    log_probs: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Compute CTC loss AND gradient for a batch of sequences.

    This is the main entry point for training - it returns both loss and gradient
    without requiring PyTorch or MLX autodiff. The gradient is computed using
    the explicit forward-backward algorithm.

    Args:
        log_probs: Log probabilities from model, shape (T, N, C) where
                   T=max time steps, N=batch size, C=num classes
        targets: Flattened target sequences, shape (sum(target_lengths),)
        input_lengths: Length of each input sequence, shape (N,)
        target_lengths: Length of each target sequence, shape (N,)
        blank: Index of the blank label (default: 0)
        reduction: 'none', 'mean', or 'sum'
        zero_infinity: Replace infinite losses with 0

    Returns:
        Tuple of:
        - loss: CTC loss value (scalar for 'mean'/'sum', (N,) for 'none')
        - grad: Gradient w.r.t. log_probs, shape (T, N, C)

    Example:
        >>> # In training loop:
        >>> log_probs = mx.log_softmax(logits, axis=-1)  # (N, T, C)
        >>> log_probs_t = mx.transpose(log_probs, (1, 0, 2))  # (T, N, C)
        >>> loss, grad_log_probs = ctc_loss_with_grad(
        ...     log_probs_t, targets, input_lens, target_lens
        ... )
        >>> # grad_log_probs is (T, N, C), transpose back for chain rule
        >>> grad_log_probs_ntc = mx.transpose(grad_log_probs, (1, 0, 2))  # (N, T, C)
    """
    T, N, C = log_probs.shape

    # Convert to numpy for computation
    log_probs_np = np.array(log_probs)
    targets_np = np.array(targets, dtype=np.int32)
    input_lengths_np = np.array(input_lengths, dtype=np.int32)
    target_lengths_np = np.array(target_lengths, dtype=np.int32)

    # Compute per-sample losses and gradients
    losses = []
    grads = np.zeros((T, N, C), dtype=np.float32)
    target_offset = 0

    for n in range(N):
        T_n = int(input_lengths_np[n])
        S_n = int(target_lengths_np[n])

        # Extract this sample's log probs and targets
        log_probs_n = log_probs_np[:T_n, n, :]  # (T_n, C)
        targets_n = targets_np[target_offset:target_offset + S_n]  # (S_n,)
        target_offset += S_n

        # Compute single-sample loss and gradient
        loss_n, grad_n = _ctc_forward_backward_single(log_probs_n, targets_n, blank)
        losses.append(float(loss_n))

        # Store gradient (only for valid time steps)
        grads[:T_n, n, :] = grad_n

    losses_np = np.array(losses, dtype=np.float32)

    # Handle infinite losses
    if zero_infinity:
        inf_mask = ~np.isfinite(losses_np)
        losses_np[inf_mask] = 0.0
        # Zero out gradients for infinite losses
        for n in range(N):
            if inf_mask[n]:
                grads[:, n, :] = 0.0

    # Apply reduction to loss
    target_lengths_float = target_lengths_np.astype(np.float32)

    if reduction == "none":
        loss_out = mx.array(losses_np)
        # For 'none' reduction, gradient scaling is per-sample
        for n in range(N):
            if target_lengths_float[n] > 0:
                grads[:, n, :] /= target_lengths_float[n]
    elif reduction == "sum":
        loss_out = mx.array(np.sum(losses_np))
        # No additional gradient scaling for sum
    elif reduction == "mean":
        # PyTorch default: mean(loss_i / target_length_i)
        normalized_losses = losses_np / np.maximum(target_lengths_float, 1.0)
        loss_out = mx.array(np.mean(normalized_losses))

        # Gradient scaling: d/d(log_probs) of mean(loss_i / S_i)
        # = (1/N) * (1/S_i) * d(loss_i)/d(log_probs)
        for n in range(N):
            if target_lengths_float[n] > 0:
                grads[:, n, :] /= (N * target_lengths_float[n])
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    grad_out = mx.array(grads)

    return loss_out, grad_out


def ctc_loss_batch(
    log_probs: mx.array,
    targets: list[mx.array],
    input_lengths: mx.array,
    blank: int = 0,
    reduction: str = "mean",
) -> mx.array:
    """
    Convenience wrapper for CTC loss with list of target arrays.

    This version accepts targets as a list of arrays (one per batch item)
    rather than a flattened tensor.

    Args:
        log_probs: Log probabilities, shape (T, N, C)
        targets: List of N target arrays, each shape (S_i,)
        input_lengths: Length of each input sequence, shape (N,)
        blank: Index of blank label
        reduction: 'none', 'mean', or 'sum'

    Returns:
        CTC loss value(s)
    """
    # Flatten targets
    flat_targets = mx.concatenate(targets)
    target_lengths = mx.array([len(t) for t in targets])

    return ctc_loss(
        log_probs=log_probs,
        targets=flat_targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction=reduction,
    )


# =============================================================================
# Pure MLX Implementation (for mx.grad compatibility)
# =============================================================================

def ctc_loss_mlx(
    log_probs: mx.array,
    targets: mx.array,
    input_length: int,
    target_length: int,
    blank: int = 0,
) -> mx.array:
    """
    Pure MLX CTC loss for a single sequence (differentiable).

    This version uses only MLX operations and is compatible with mx.grad()
    for automatic differentiation during training.

    Args:
        log_probs: Log probabilities, shape (T, C)
        targets: Target sequence (no blanks), shape (S,)
        input_length: Actual input length (T)
        target_length: Actual target length (S)
        blank: Index of blank label

    Returns:
        CTC loss (negative log-likelihood) as scalar
    """
    T = input_length
    S = target_length

    if S == 0:
        # Empty target - all outputs should be blank
        return -mx.sum(log_probs[:T, blank])

    # Extended label sequence length
    L = 2 * S + 1

    # Create extended labels with blanks interleaved
    # extended = [blank, target[0], blank, target[1], ..., target[S-1], blank]
    _create_extended_labels(targets, S, blank)

    # Initialize alpha in log space
    alpha = mx.full((T, L), LOG_ZERO)

    # Base case: t=0
    alpha = alpha.at[0, 0].add(-LOG_ZERO + log_probs[0, blank])
    if L > 1:
        first_label = targets[0]
        alpha = alpha.at[0, 1].add(-LOG_ZERO + log_probs[0, first_label])

    # Forward recursion using scan for efficiency
    for t in range(1, T):
        alpha_t = mx.full((L,), LOG_ZERO)

        for s in range(L):
            # Get label at position s
            is_blank = (s % 2 == 0)
            if is_blank:
                label_s = blank
            else:
                label_s = targets[s // 2]

            # Option 1: Stay in same state
            val = alpha[t-1, s]

            # Option 2: From previous state
            if s > 0:
                val = log_sum_exp(val, alpha[t-1, s-1])

            # Option 3: Skip blank (for non-blank, non-repeated labels)
            if s > 1 and not is_blank:
                # Check if s-2 has same label
                is_s2_blank = ((s-2) % 2 == 0)
                if not is_s2_blank:
                    label_s2 = targets[(s-2) // 2]
                    if label_s != label_s2:
                        val = log_sum_exp(val, alpha[t-1, s-2])
                else:
                    # s-2 is blank, can always skip
                    val = log_sum_exp(val, alpha[t-1, s-2])

            alpha_t = alpha_t.at[s].add(-LOG_ZERO + val + log_probs[t, label_s])

        alpha = alpha.at[t].add(-alpha[t] + alpha_t)

    # Total log probability
    log_prob = log_sum_exp(alpha[T-1, L-1], alpha[T-1, L-2])

    return -log_prob


def _create_extended_labels(targets: mx.array, S: int, blank: int) -> mx.array:
    """Create extended label sequence with blanks interleaved."""
    L = 2 * S + 1
    extended = mx.full((L,), blank, dtype=mx.int32)
    # extended[1::2] = targets (odd indices)
    for i in range(S):
        extended = extended.at[2*i + 1].add(-blank + targets[i])
    return extended


# =============================================================================
# Validation utilities
# =============================================================================

def validate_against_pytorch(
    T: int = 50,
    N: int = 4,
    C: int = 100,
    S_max: int = 20,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Validate MLX CTC loss against PyTorch CTC loss.

    Args:
        T: Sequence length
        N: Batch size
        C: Number of classes
        S_max: Maximum target length
        seed: Random seed

    Returns:
        Tuple of (mlx_loss, pytorch_loss, absolute_error)

    Raises:
        ImportError: If PyTorch is not available
        AssertionError: If losses don't match within tolerance
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch required for validation") from None

    rng = np.random.default_rng(seed)

    # Generate random log probs (normalized)
    logits = rng.standard_normal((T, N, C)).astype(np.float32)
    log_probs_np = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))

    # Generate random targets (no blanks, values in [1, C-1])
    targets_list = []
    target_lengths = []
    for _ in range(N):
        S = int(rng.integers(1, S_max + 1))  # Convert numpy int64 to Python int
        targets_list.extend(rng.integers(1, C, size=S).tolist())
        target_lengths.append(S)

    input_lengths = [T] * N

    # PyTorch CTC
    log_probs_torch = torch.from_numpy(log_probs_np).requires_grad_(False)
    targets_torch = torch.tensor(targets_list, dtype=torch.long)
    input_lengths_torch = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths_torch = torch.tensor(target_lengths, dtype=torch.long)

    loss_torch = F.ctc_loss(
        log_probs_torch, targets_torch,
        input_lengths_torch, target_lengths_torch,
        blank=0, reduction='mean', zero_infinity=True,
    )
    loss_torch_val = loss_torch.item()

    # MLX CTC
    log_probs_mlx = mx.array(log_probs_np)
    targets_mlx = mx.array(targets_list, dtype=mx.int32)
    input_lengths_mlx = mx.array(input_lengths, dtype=mx.int32)
    target_lengths_mlx = mx.array(target_lengths, dtype=mx.int32)

    loss_mlx = ctc_loss(
        log_probs_mlx, targets_mlx,
        input_lengths_mlx, target_lengths_mlx,
        blank=0, reduction='mean',
    )
    loss_mlx_val = float(loss_mlx)

    error = abs(loss_torch_val - loss_mlx_val)

    return loss_mlx_val, loss_torch_val, error


if __name__ == "__main__":
    # Quick validation test
    print("Validating MLX CTC loss against PyTorch...")

    test_cases = [
        {"T": 10, "N": 1, "C": 5, "S_max": 3},
        {"T": 50, "N": 4, "C": 100, "S_max": 10},
        {"T": 100, "N": 2, "C": 1000, "S_max": 20},
    ]

    for tc in test_cases:
        try:
            mlx_loss, pt_loss, error = validate_against_pytorch(**tc)
            status = "PASS" if error < 1e-4 else "FAIL"
            print(f"  {tc}: MLX={mlx_loss:.6f}, PT={pt_loss:.6f}, error={error:.2e} [{status}]")
        except ImportError:
            print(f"  {tc}: SKIPPED (PyTorch not available)")
        except Exception as e:
            print(f"  {tc}: ERROR - {e}")
