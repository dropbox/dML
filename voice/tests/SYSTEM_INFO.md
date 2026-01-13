# System Information

**Date**: 2025-11-24
**Phase**: Phase 1 - Python Prototype
**Status**: Environment Setup Complete ✅

## Hardware Specifications

### M4 Max Chip
- **Model**: Apple M4 Max
- **GPU Cores**: 40 cores
- **Metal Support**: Metal 3
- **CPU Cores**: 16 physical / 16 logical
- **Unified Memory**: 128 GB (137,438,953,472 bytes)

### Memory Details
- Total System Memory: 128 GB
- Unified Memory Architecture (shared between CPU and GPU)
- High bandwidth memory access for GPU operations

## Software Environment

### Python Environment
- **Python Version**: 3.11.14
- **Virtual Environment**: venv (located at /Users/ayates/voice/venv)
- **Installation Method**: Homebrew python@3.11

### PyTorch Configuration
- **PyTorch Version**: 2.9.1
- **MPS (Metal Performance Shaders) Available**: Yes
- **MPS Built**: Yes
- **CUDA Available**: No (not needed on Apple Silicon)

### Key Dependencies Installed
- `torch==2.9.1` - Deep learning framework with Metal support
- `torchvision==0.24.1` - Computer vision utilities
- `torchaudio==2.9.1` - Audio processing utilities
- `transformers==4.57.2` - Hugging Face transformers library
- `accelerate==1.12.0` - Training acceleration library
- `TTS==0.22.0` - Coqui TTS for speech synthesis
- `sounddevice==0.5.3` - Audio playback library
- `pyaudio==0.2.14` - Audio I/O library
- `numpy==1.26.4` - Numerical computing (downgraded for TTS compatibility)
- `scipy==1.16.3` - Scientific computing
- `librosa==0.11.0` - Audio analysis
- `safetensors==0.7.0` - Safe tensor serialization
- `portaudio==19.7.0` - Audio backend (via Homebrew)

### Metal GPU Capabilities
- **Device**: mps (Metal Performance Shaders)
- **Native bfloat16 Support**: Yes (M4 native format)
- **Maximum Performance Mode**: Available via torch.compile()
- **Unified Memory Access**: Direct GPU-CPU memory sharing

## Performance Expectations

Based on M4 Max specifications:

### Translation (NLLB-200-3.3B)
- **Target**: < 30ms per sentence
- **Model Size**: 3.3 billion parameters
- **Precision**: bfloat16 (native M4 format)
- **Expected GPU Utilization**: 70-90%

### TTS (XTTS v2)
- **Target**: < 80ms per sentence
- **Precision**: float32 (to be optimized to bfloat16)
- **Expected GPU Utilization**: 80-95%

### Total Pipeline
- **Phase 1 Target**: < 150ms end-to-end latency
- **Phase 2 Target**: < 70ms (with C++ optimization)
- **Phase 3 Target**: < 50ms (with custom Metal kernels)

## Notes

- Successfully installed all dependencies including Coqui TTS 0.22.0
- Resolved Python version conflict by using Python 3.11.14 (TTS requires < 3.12)
- NumPy downgraded to 1.26.4 for TTS compatibility (from 2.3.5)
- All dependencies tested and verified compatible
- Metal GPU is operational and ready for Phase 1 implementation

## Next Steps

1. Download NLLB-200-3.3B model and test inference speed
2. Download XTTS v2 model and test synthesis speed
3. Benchmark single-inference latency on Metal GPU
4. Proceed to prototype_pipeline.py implementation
