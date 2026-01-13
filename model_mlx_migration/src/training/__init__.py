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
Training infrastructure for Zipformer ASR model.

Phase 4 of SOTA++ Voice Server STT implementation.
"""

from .config import (
    OptimizerConfig,
    RichAudioHeadsTrainingConfig,
    SchedulerConfig,
    SpeakerTrainingConfig,
    TrainingConfig,
)
from .dataloader import (
    ASRDataLoader,
    AudioSample,
    LibriSpeechDataset,
    compute_fbank_features,
    create_librispeech_loader,
    load_audio_file,
)
from .loss import (
    CRCTCLoss,
    CTCLoss,
    RichAudioLoss,
    RichAudioLossOutput,
    TransducerLoss,
    binary_cross_entropy_loss,
    cr_ctc_loss,
    # Rich audio loss functions
    cross_entropy_loss,
    ctc_loss,
    l1_loss,
    mse_loss,
    transducer_loss_simple,
)
from .rich_audio_dataloader import (
    EMOTION_LABELS,
    MELD_EMOTION_TO_INT,
    PARALINGUISTIC_LABELS,
    CombinedEmotionDataset,
    CREMADDataset,
    MELDDataset,
    RichAudioDataLoader,
    RichAudioSample,
    VocalSetDataset,
    VocalSoundDataset,
    create_combined_emotion_loader,
    create_emotion_loader,
    create_meld_loader,
    create_paralinguistics_loader,
    create_pitch_loader,
)
from .scheduler import WarmupScheduler, get_scheduler
from .speaker_dataloader import (
    CNCelebDataset,
    CombinedSpeakerDataset,
    LibriSpeechSpeakerDataset,
    SpeakerDataLoader,
    SpeakerDataset,
    SpeakerSample,
    VerificationTrialLoader,
    create_cnceleb_loader,
    create_librispeech_speaker_loader,
)
from .speaker_trainer import (
    SpeakerTrainer,
    SpeakerTrainingMetrics,
    SpeakerTrainingState,
    create_speaker_datasets,
    create_speaker_trainer,
)
from .trainer import Trainer, TrainingState

__all__ = [
    # Config
    "TrainingConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "RichAudioHeadsTrainingConfig",
    "SpeakerTrainingConfig",
    # Loss functions
    "transducer_loss_simple",
    "ctc_loss",
    "cr_ctc_loss",
    "TransducerLoss",
    "CTCLoss",
    "CRCTCLoss",
    # Rich audio loss functions
    "cross_entropy_loss",
    "binary_cross_entropy_loss",
    "mse_loss",
    "l1_loss",
    "RichAudioLoss",
    "RichAudioLossOutput",
    # Scheduler
    "WarmupScheduler",
    "get_scheduler",
    # Trainer
    "Trainer",
    "TrainingState",
    # Data loading
    "AudioSample",
    "LibriSpeechDataset",
    "ASRDataLoader",
    "load_audio_file",
    "compute_fbank_features",
    "create_librispeech_loader",
    # Speaker embedding training
    "SpeakerSample",
    "SpeakerDataset",
    "CNCelebDataset",
    "LibriSpeechSpeakerDataset",
    "CombinedSpeakerDataset",
    "SpeakerDataLoader",
    "VerificationTrialLoader",
    "create_cnceleb_loader",
    "create_librispeech_speaker_loader",
    "SpeakerTrainer",
    "SpeakerTrainingState",
    "SpeakerTrainingMetrics",
    "create_speaker_trainer",
    "create_speaker_datasets",
    # Rich audio head training
    "RichAudioSample",
    "CREMADDataset",
    "VocalSetDataset",
    "CombinedEmotionDataset",
    "VocalSoundDataset",
    "MELDDataset",
    "RichAudioDataLoader",
    "create_emotion_loader",
    "create_pitch_loader",
    "create_combined_emotion_loader",
    "create_paralinguistics_loader",
    "create_meld_loader",
    "EMOTION_LABELS",
    "PARALINGUISTIC_LABELS",
    "MELD_EMOTION_TO_INT",
]
