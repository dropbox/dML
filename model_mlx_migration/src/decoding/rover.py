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
ROVER: Recognizer Output Voting Error Reduction.

Implementation of ROVER voting algorithm for combining multiple ASR
hypotheses into a single, more accurate output.

Reference: Fiscus, J. G. (1997). A post-processing system to yield
reduced word error rates: Recognizer output voting error reduction (ROVER).

ROVER works by:
1. Aligning multiple hypotheses into a word transition network (WTN)
2. Voting on the best word at each position
3. Using confidence scores to weight votes (optional)

This implementation supports:
- Multiple hypothesis sources (Transducer, CTC, Whisper, etc.)
- Confidence-weighted voting
- Phoneme-based mismatch weighting (SOTA++ addition)
- Configurable null word handling

Usage:
    hypotheses = [
        Hypothesis(words=["hello", "world"], confidences=[0.9, 0.85]),
        Hypothesis(words=["hello", "word"], confidences=[0.8, 0.75]),
        Hypothesis(words=["halo", "world"], confidences=[0.7, 0.9]),
    ]

    rover = ROVER()
    result = rover.vote(hypotheses)
    print(result.words)  # ["hello", "world"]
"""

from dataclasses import dataclass
from enum import Enum

# Special null token for deletions
NULL_TOKEN = "<NULL>"


class AlignmentMethod(Enum):
    """Methods for aligning hypotheses."""

    SIMPLE = "simple"  # Position-based alignment (fast, less accurate)
    DP = "dp"  # Dynamic programming alignment (slower, more accurate)


@dataclass
class ROVERConfig:
    """Configuration for ROVER voting."""

    # Alignment method
    alignment_method: AlignmentMethod = AlignmentMethod.DP

    # Confidence weighting
    use_confidence_weights: bool = True
    min_confidence: float = 0.01  # Floor for confidence scores

    # Null word handling
    null_weight: float = 0.5  # Weight for null votes (deletions)
    prefer_non_null: bool = True  # Tie-break in favor of words over null

    # Phoneme weighting (SOTA++)
    use_phoneme_weighting: bool = False
    phoneme_mismatch_penalty: float = 0.2  # Penalty for phoneme disagreement

    # Voting parameters
    min_votes_to_emit: int = 1  # Minimum votes needed to emit a word
    tie_break_by_confidence: bool = True  # Use confidence for tie-breaking


@dataclass
class Hypothesis:
    """A single ASR hypothesis with optional confidence scores."""

    words: list[str]
    confidences: list[float] | None = None
    source: str = "unknown"  # e.g., "transducer", "ctc", "whisper"

    # Optional phoneme sequences for phoneme weighting
    phonemes: list[list[str]] | None = None

    def __post_init__(self):
        """Validate and fill in defaults."""
        if self.confidences is None:
            self.confidences = [1.0] * len(self.words)
        elif len(self.confidences) != len(self.words):
            msg = (
                f"Length mismatch: {len(self.words)} words vs "
                f"{len(self.confidences)} confidences"
            )
            raise ValueError(msg)
        if self.phonemes is not None and len(self.phonemes) != len(self.words):
            msg = (
                f"Length mismatch: {len(self.words)} words vs "
                f"{len(self.phonemes)} phoneme sequences"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.words)


@dataclass
class AlignedHypothesis:
    """Hypothesis aligned to the word transition network."""

    words: list[str]  # May contain NULL_TOKEN
    confidences: list[float]
    phonemes: list[list[str] | None]
    original: Hypothesis
    alignment_indices: list[int]  # Index in WTN for each word


@dataclass
class VotingSlot:
    """A single position in the word transition network."""

    position: int
    candidates: dict[str, float]  # word -> total weighted vote
    vote_counts: dict[str, int]  # word -> number of votes
    sources: dict[str, list[str]]  # word -> list of sources that voted for it


@dataclass
class ROVERResult:
    """Result of ROVER voting."""

    words: list[str]
    confidences: list[float]
    slots: list[VotingSlot]  # Detailed voting information

    @property
    def text(self) -> str:
        """Get result as space-separated string."""
        return " ".join(self.words)

    @property
    def word_count(self) -> int:
        return len(self.words)


def _simple_align(
    hypotheses: list[Hypothesis],
    config: ROVERConfig,
) -> tuple[list[AlignedHypothesis], int]:
    """
    Simple position-based alignment.

    Pads shorter hypotheses with NULL_TOKEN.
    """
    max_len = max(len(h) for h in hypotheses)

    aligned = []
    for hyp in hypotheses:
        words = list(hyp.words)
        confs = list(hyp.confidences)
        phs: list[list[str] | None]
        if hyp.phonemes is None:
            phs = [None] * len(words)
        else:
            phs = [list(p) for p in hyp.phonemes]

        # Pad with null
        while len(words) < max_len:
            words.append(NULL_TOKEN)
            confs.append(config.null_weight)
            phs.append(None)

        aligned.append(
            AlignedHypothesis(
                words=words,
                confidences=confs,
                phonemes=phs,
                original=hyp,
                alignment_indices=list(range(max_len)),
            ),
        )

    return aligned, max_len


def _dp_edit_distance(
    seq1: list[str],
    seq2: list[str],
) -> tuple[int, list[tuple[int, int]]]:
    """
    Compute edit distance and alignment between two sequences.

    Returns:
        Tuple of (edit_distance, alignment)
        alignment is list of (idx1, idx2) pairs, -1 means gap
    """
    n, m = len(seq1), len(seq2)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Delete from seq1
                    dp[i][j - 1],  # Insert into seq1
                    dp[i - 1][j - 1],  # Substitute
                )

    # Backtrack to get alignment
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and (
            seq1[i - 1] == seq2[j - 1] or dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((i - 1, -1))  # Deletion
            i -= 1
        else:
            alignment.append((-1, j - 1))  # Insertion
            j -= 1

    alignment.reverse()
    return dp[n][m], alignment


def _dp_align(
    hypotheses: list[Hypothesis],
    config: ROVERConfig,
) -> tuple[list[AlignedHypothesis], int]:
    """
    Dynamic programming based alignment using edit distance.

    Uses the first hypothesis as reference and aligns others to it.
    """
    if len(hypotheses) == 0:
        return [], 0

    if len(hypotheses) == 1:
        hyp = hypotheses[0]
        return [
            AlignedHypothesis(
                words=list(hyp.words),
                confidences=list(hyp.confidences),
                phonemes=[list(p) for p in hyp.phonemes] if hyp.phonemes else [None] * len(hyp.words),
                original=hyp,
                alignment_indices=list(range(len(hyp.words))),
            ),
        ], len(hyp.words)

    # Use first hypothesis as reference (could use median length instead)
    reference = hypotheses[0]

    # Build word transition network iteratively
    # Start with reference
    wtn_positions: list[set[str]] = [{word} for word in reference.words]

    aligned_list = []

    # Align reference to itself
    aligned_list.append(
        AlignedHypothesis(
            words=list(reference.words),
            confidences=list(reference.confidences),
            phonemes=[list(p) for p in reference.phonemes] if reference.phonemes else [None] * len(reference.words),
            original=reference,
            alignment_indices=list(range(len(reference.words))),
        ),
    )

    # Align each other hypothesis to the growing WTN
    current_reference = list(reference.words)

    for hyp in hypotheses[1:]:
        _, alignment = _dp_edit_distance(current_reference, list(hyp.words))

        # Build aligned version
        aligned_words = []
        aligned_confs = []
        aligned_indices = []
        aligned_phonemes: list[list[str] | None] = []


        new_wtn_positions = []

        for ref_pos, hyp_pos in alignment:
            if ref_pos >= 0 and hyp_pos >= 0:
                # Match or substitution
                if ref_pos < len(wtn_positions):
                    new_wtn_positions.append(
                        wtn_positions[ref_pos] | {hyp.words[hyp_pos]},
                    )
                else:
                    new_wtn_positions.append({current_reference[ref_pos], hyp.words[hyp_pos]})
                aligned_words.append(hyp.words[hyp_pos])
                aligned_confs.append(hyp.confidences[hyp_pos])
                if hyp.phonemes is not None:
                    aligned_phonemes.append(list(hyp.phonemes[hyp_pos]))
                else:
                    aligned_phonemes.append(None)
                aligned_indices.append(len(new_wtn_positions) - 1)
            elif ref_pos >= 0:
                # Deletion (word in reference but not in hypothesis)
                if ref_pos < len(wtn_positions):
                    new_wtn_positions.append(wtn_positions[ref_pos] | {NULL_TOKEN})
                else:
                    new_wtn_positions.append({current_reference[ref_pos], NULL_TOKEN})
                aligned_words.append(NULL_TOKEN)
                aligned_confs.append(config.null_weight)
                aligned_phonemes.append(None)
                aligned_indices.append(len(new_wtn_positions) - 1)
            else:
                # Insertion (word in hypothesis but not in reference)
                new_wtn_positions.append({NULL_TOKEN, hyp.words[hyp_pos]})
                aligned_words.append(hyp.words[hyp_pos])
                aligned_confs.append(hyp.confidences[hyp_pos])
                if hyp.phonemes is not None:
                    aligned_phonemes.append(list(hyp.phonemes[hyp_pos]))
                else:
                    aligned_phonemes.append(None)
                aligned_indices.append(len(new_wtn_positions) - 1)

        aligned_list.append(
            AlignedHypothesis(
                words=aligned_words,
                confidences=aligned_confs,
                phonemes=aligned_phonemes,
                original=hyp,
                alignment_indices=aligned_indices,
            ),
        )

        wtn_positions = new_wtn_positions

    # Re-align first hypothesis to final WTN length
    wtn_length = len(wtn_positions)
    if len(aligned_list[0].words) < wtn_length:
        # Need to extend with nulls
        first_aligned = aligned_list[0]
        extended_words = list(first_aligned.words)
        extended_confs = list(first_aligned.confidences)
        extended_phonemes = (
            list(first_aligned.phonemes)
            if getattr(first_aligned, "phonemes", None) is not None
            else [None] * len(extended_words)
        )
        extended_indices = list(first_aligned.alignment_indices)

        while len(extended_words) < wtn_length:
            extended_words.append(NULL_TOKEN)
            extended_confs.append(config.null_weight)
            extended_phonemes.append(None)
            extended_indices.append(len(extended_words) - 1)

        aligned_list[0] = AlignedHypothesis(
            words=extended_words,
            confidences=extended_confs,
            phonemes=extended_phonemes,
            original=first_aligned.original,
            alignment_indices=extended_indices,
        )

    return aligned_list, wtn_length


def align_hypotheses(
    hypotheses: list[Hypothesis],
    config: ROVERConfig | None = None,
) -> tuple[list[AlignedHypothesis], int]:
    """
    Align multiple hypotheses into a word transition network.

    Args:
        hypotheses: List of ASR hypotheses to align
        config: ROVER configuration

    Returns:
        Tuple of (aligned_hypotheses, wtn_length)
    """
    if config is None:
        config = ROVERConfig()

    if len(hypotheses) == 0:
        return [], 0

    if config.alignment_method == AlignmentMethod.SIMPLE:
        return _simple_align(hypotheses, config)
    return _dp_align(hypotheses, config)


def _compute_phoneme_similarity(
    phonemes1: list[str] | None,
    phonemes2: list[str] | None,
) -> float:
    """
    Compute similarity between two phoneme sequences.

    Returns 1.0 for identical, 0.0 for completely different.
    """
    if phonemes1 is None or phonemes2 is None:
        return 1.0  # No penalty if phonemes not available

    if len(phonemes1) == 0 and len(phonemes2) == 0:
        return 1.0

    if len(phonemes1) == 0 or len(phonemes2) == 0:
        return 0.0

    # Simple overlap measure
    set1 = set(phonemes1)
    set2 = set(phonemes2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def vote_rover(
    hypotheses: list[Hypothesis],
    config: ROVERConfig | None = None,
) -> ROVERResult:
    """
    Perform ROVER voting on multiple ASR hypotheses.

    Args:
        hypotheses: List of ASR hypotheses to combine
        config: ROVER configuration

    Returns:
        ROVERResult with voted words and metadata
    """
    if config is None:
        config = ROVERConfig()

    if len(hypotheses) == 0:
        return ROVERResult(words=[], confidences=[], slots=[])

    if len(hypotheses) == 1:
        hyp = hypotheses[0]
        slots = [
            VotingSlot(
                position=i,
                candidates={w: c},
                vote_counts={w: 1},
                sources={w: [hyp.source]},
            )
            for i, (w, c) in enumerate(zip(hyp.words, hyp.confidences, strict=False))
        ]
        return ROVERResult(
            words=list(hyp.words),
            confidences=list(hyp.confidences),
            slots=slots,
        )

    # Align hypotheses
    aligned, wtn_length = align_hypotheses(hypotheses, config)

    # Create voting slots
    slots: list[VotingSlot] = []

    for pos in range(wtn_length):
        candidates: dict[str, float] = {}
        vote_counts: dict[str, int] = {}
        sources: dict[str, list[str]] = {}
        slot_votes: list[tuple[str, float, list[str] | None]] = []  # (word, weight, phonemes)

        for ah in aligned:
            if pos < len(ah.words):
                word = ah.words[pos]
                conf = ah.confidences[pos]
                source = ah.original.source
                phonemes = ah.phonemes[pos] if pos < len(ah.phonemes) else None

                # Apply confidence floor
                conf = max(conf, config.min_confidence)

                # Weight calculation
                if config.use_confidence_weights:
                    weight = conf
                else:
                    weight = 1.0

                # Apply null weight
                if word == NULL_TOKEN:
                    weight *= config.null_weight

                # Accumulate votes
                if word not in candidates:
                    candidates[word] = 0.0
                    vote_counts[word] = 0
                    sources[word] = []

                candidates[word] += weight
                vote_counts[word] += 1
                sources[word].append(source)
                slot_votes.append((word, weight, phonemes))

        if config.use_phoneme_weighting and slot_votes:
            votes_with_phonemes = [
                (w, wt, ph)
                for (w, wt, ph) in slot_votes
                if w != NULL_TOKEN and ph is not None
            ]
            total_ph_weight = sum(wt for (_, wt, _) in votes_with_phonemes)

            if total_ph_weight > 0.0:
                candidate_phonemes: dict[str, list[str]] = {}
                for w, _, ph in votes_with_phonemes:
                    if ph is None:
                        continue
                    if w not in candidate_phonemes:
                        candidate_phonemes[w] = list(ph)
                    else:
                        # Union to handle minor tokenization differences across sources.
                        seen = set(candidate_phonemes[w])
                        for p in ph:
                            if p not in seen:
                                candidate_phonemes[w].append(p)
                                seen.add(p)

                for word in list(candidates.keys()):
                    if word == NULL_TOKEN:
                        continue
                    if word not in candidate_phonemes:
                        continue

                    ph_word = candidate_phonemes[word]
                    agreement = 0.0
                    for _, wt, ph_other in votes_with_phonemes:
                        agreement += wt * _compute_phoneme_similarity(ph_word, ph_other)

                    agreement /= total_ph_weight
                    factor = 1.0 - config.phoneme_mismatch_penalty * (1.0 - agreement)
                    candidates[word] *= max(factor, 0.0)

        slots.append(
            VotingSlot(
                position=pos,
                candidates=candidates,
                vote_counts=vote_counts,
                sources=sources,
            ),
        )

    # Select best word at each position
    result_words = []
    result_confidences = []

    for slot in slots:
        if not slot.candidates:
            continue

        # Get best candidate
        best_word = None
        best_score = -1.0

        for word, score in slot.candidates.items():
            vote_count = slot.vote_counts[word]

            # Skip if below minimum votes
            if vote_count < config.min_votes_to_emit:
                continue

            is_better = False

            if score > best_score:
                is_better = True
            elif score == best_score and config.tie_break_by_confidence:
                # Tie-break by vote count
                if vote_count > slot.vote_counts.get(best_word, 0):
                    is_better = True
                # Prefer non-null in ties
                elif config.prefer_non_null and word != NULL_TOKEN and best_word == NULL_TOKEN:
                    is_better = True

            if is_better:
                best_word = word
                best_score = score

        # Emit word if it's not null
        if best_word is not None and best_word != NULL_TOKEN:
            result_words.append(best_word)
            # Normalize confidence
            total_votes = sum(slot.vote_counts.values())
            normalized_conf = best_score / (total_votes if total_votes > 0 else 1)
            result_confidences.append(min(normalized_conf, 1.0))

    return ROVERResult(
        words=result_words,
        confidences=result_confidences,
        slots=slots,
    )


class ROVER:
    """
    ROVER voting combiner for multiple ASR hypotheses.

    This class provides a stateful interface to ROVER voting,
    supporting source registration and phoneme weighting.
    """

    def __init__(self, config: ROVERConfig | None = None):
        """
        Initialize ROVER combiner.

        Args:
            config: ROVER configuration
        """
        self.config = config or ROVERConfig()
        self.sources: dict[str, float] = {}  # source -> weight multiplier

    def register_source(self, name: str, weight: float = 1.0) -> None:
        """
        Register an ASR source with optional weight.

        Args:
            name: Source identifier (e.g., "transducer", "whisper")
            weight: Weight multiplier for this source's votes
        """
        self.sources[name] = weight

    def vote(
        self,
        hypotheses: list[Hypothesis],
        phoneme_sequences: dict[str, list[list[str]]] | None = None,
    ) -> ROVERResult:
        """
        Perform ROVER voting on hypotheses.

        Args:
            hypotheses: List of ASR hypotheses
            phoneme_sequences: Optional dict mapping source -> phonemes per word

        Returns:
            ROVERResult with voted output
        """
        # Apply source weights
        weighted_hypotheses = []
        for hyp in hypotheses:
            source_weight = self.sources.get(hyp.source, 1.0)
            weighted_confs = [c * source_weight for c in hyp.confidences]
            weighted_hypotheses.append(
                Hypothesis(
                    words=hyp.words,
                    confidences=weighted_confs,
                    source=hyp.source,
                    phonemes=phoneme_sequences.get(hyp.source) if phoneme_sequences else None,
                ),
            )

        return vote_rover(weighted_hypotheses, self.config)

    def combine_transducer_ctc(
        self,
        transducer_words: list[str],
        transducer_confs: list[float],
        ctc_words: list[str],
        ctc_confs: list[float],
    ) -> ROVERResult:
        """
        Combine transducer and CTC hypotheses.

        This is a common use case for Zipformer which has both heads.

        Args:
            transducer_words: Words from transducer decoder
            transducer_confs: Confidence scores from transducer
            ctc_words: Words from CTC decoder
            ctc_confs: Confidence scores from CTC

        Returns:
            ROVERResult with combined output
        """
        hypotheses = [
            Hypothesis(words=transducer_words, confidences=transducer_confs, source="transducer"),
            Hypothesis(words=ctc_words, confidences=ctc_confs, source="ctc"),
        ]
        return self.vote(hypotheses)

    def combine_with_whisper(
        self,
        primary_words: list[str],
        primary_confs: list[float],
        primary_source: str,
        whisper_words: list[str],
        whisper_confs: list[float],
    ) -> ROVERResult:
        """
        Combine primary ASR with Whisper fallback.

        Args:
            primary_words: Words from primary ASR (e.g., Zipformer)
            primary_confs: Confidence scores from primary
            primary_source: Source name for primary
            whisper_words: Words from Whisper
            whisper_confs: Confidence scores from Whisper

        Returns:
            ROVERResult with combined output
        """
        hypotheses = [
            Hypothesis(words=primary_words, confidences=primary_confs, source=primary_source),
            Hypothesis(words=whisper_words, confidences=whisper_confs, source="whisper"),
        ]
        return self.vote(hypotheses)

    def combine_three_way(
        self,
        transducer_words: list[str],
        transducer_confs: list[float],
        ctc_words: list[str],
        ctc_confs: list[float],
        whisper_words: list[str],
        whisper_confs: list[float],
    ) -> ROVERResult:
        """
        Combine transducer, CTC, and Whisper hypotheses.

        This is the full high-accuracy mode with three signals.

        Args:
            transducer_words: Words from transducer decoder
            transducer_confs: Confidence scores from transducer
            ctc_words: Words from CTC decoder
            ctc_confs: Confidence scores from CTC
            whisper_words: Words from Whisper
            whisper_confs: Confidence scores from Whisper

        Returns:
            ROVERResult with combined output
        """
        hypotheses = [
            Hypothesis(words=transducer_words, confidences=transducer_confs, source="transducer"),
            Hypothesis(words=ctc_words, confidences=ctc_confs, source="ctc"),
            Hypothesis(words=whisper_words, confidences=whisper_confs, source="whisper"),
        ]
        return self.vote(hypotheses)


def compute_oracle_wer(
    hypotheses: list[Hypothesis],
    reference: list[str],
) -> tuple[float, Hypothesis]:
    """
    Compute oracle WER - the best possible WER from any hypothesis.

    This is useful for understanding the upper bound of ROVER performance.

    Args:
        hypotheses: List of ASR hypotheses
        reference: Reference transcription

    Returns:
        Tuple of (best_wer, best_hypothesis)
    """
    if not hypotheses:
        return 1.0, Hypothesis(words=[], confidences=[])

    best_wer = float("inf")
    best_hyp = hypotheses[0]

    for hyp in hypotheses:
        # Compute WER via edit distance
        distance, _ = _dp_edit_distance(reference, list(hyp.words))
        wer = distance / len(reference) if reference else 0.0

        if wer < best_wer:
            best_wer = wer
            best_hyp = hyp

    return best_wer, best_hyp
