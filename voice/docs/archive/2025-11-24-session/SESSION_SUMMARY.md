# Voice TTS Session Summary
**Date**: 2025-11-24
**Session**: Continue command execution

---

## Current Status

###  ✅ What's Working
1. **llama.cpp**: Built successfully with Metal support for M4 Max
2. **Qwen2.5-7B-Instruct-Q3_K_M**: Model downloaded (3.5GB) and verified
3. **Translation Testing**: Qwen model successfully translates English to Japanese on Metal GPU
4. **Existing System**: Rust + Python TTS pipeline working (258ms latency)

### 🔄 What Needs Work
1. **Qwen Worker Script**: Needs flag updates for proper output (`--single-turn` flag)
2. **CosyVoice Installation**: Not yet started (next major task)
3. **Ultimate Pipeline Integration**: Pending Cosy Voice setup

---

## Technical Achievements This Session

### Qwen2.5-7B Setup
- **Model**: Q3_K_M quantization (3.5GB single file)
- **Location**: `/Users/ayates/voice/models/qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q3_k_m.gguf`
- **llama.cpp**: `/Users/ayates/voice/models/qwen/llama.cpp/build/bin/llama-cli`
- **Performance**: ~190ms translation time on Metal GPU (after model load)
- **Quality**: Produces excellent Japanese translations

### Example Translation Test
```bash
# Input: "Good morning"
# Output: "おはようございます" (Ohayou gozaimasu)
# Time: ~190ms
# GPU: Apple M4 Max with Metal acceleration
```

### Key Findings
1. **Model Loading**: First inference takes ~7 seconds (model load), subsequent inferences ~200-600ms
2. **Flags Needed**: `--single-turn --no-cnv` for proper non-interactive mode
3. **Output Parsing**: llama.cpp includes verbose output; need to filter to just translation
4. **Temperature**: 0.1-0.3 works well for consistent translations

---

## Files Created/Modified

### New Files
- `/Users/ayates/voice/download_qwen.sh` - Clean Qwen model download script
- `/Users/ayates/voice/SESSION_SUMMARY.md` - This file

### Modified Files
- `/Users/ayates/voice/stream-tts-rust/python/translation_worker_qwen_llamacpp.py`
  - Updated model path from 72B to 7B Q3_K_M
  - Still needs flag updates for `--single-turn`

---

## Next Steps (Priority Order)

### Immediate (Can do now)
1. **Update Qwen worker flags**
   - Add `--single-turn` flag to translation_worker_qwen_llamacpp.py:80
   - Improve output parsing to extract just Japanese text
   - Test end-to-end with simple input

2. **Install CosyVoice 3.0**
   - Run existing `setup_cosyvoice.sh` script
   - Download CosyVoice-300M-SFT model (~3GB)
   - Test TTS worker: `stream-tts-rust/python/tts_worker_cosyvoice.py`

3. **Test Ultimate Pipeline**
   - Run `test_ultimate_tts.sh` once both workers are ready
   - Measure end-to-end latency
   - Compare quality vs current system

### Short-term (This week)
4. **Optimize Qwen prompts**
   - Fine-tune temperature and sampling params
   - Reduce verbose output from model
   - Achieve target < 200ms translation time

5. **Production Integration**
   - Update `run_ultimate_tts.sh` to use Qwen + CosyVoice
   - Test with Claude Code output
   - Deploy for daily use

### Medium-term (Next 1-2 weeks)
6. **Performance optimization**
   - Implement model caching (keep in memory)
   - Parallel processing where possible
   - Target < 300ms end-to-end latency

7. **Phase 2: C++ Implementation**
   - Follow MASTER_ROADMAP.md Phase 2 plan
   - Build C++ coordinator with Metal
   - Target < 100ms latency

---

## Performance Targets

| System | Translation | TTS | Total | Status |
|--------|-------------|-----|-------|--------|
| **Current (Rust+Python)** | 110ms | 148ms | **258ms** | ✅ Working |
| **Ultimate (Qwen+Cosy)** | 200ms | 150ms | **350ms** | 🔄 80% complete |
| **Phase 2 (C++/Metal)** | 50ms | 50ms | **100ms** | 📝 Planned |
| **Phase 3 (Optimized)** | 20ms | 30ms | **50ms** | 🎯 Goal |

---

## Key Commands Reference

### Test Qwen Translation (Direct)
```bash
/Users/ayates/voice/models/qwen/llama.cpp/build/bin/llama-cli \
  --model /Users/ayates/voice/models/qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q3_k_m.gguf \
  --prompt "Translate to Japanese: Hello\nJapanese:" \
  --n-gpu-layers 999 \
  --temp 0.1 \
  --n-predict 20 \
  --single-turn \
  --verbosity 0
```

### Test Translation Worker
```bash
echo "Hello, how are you?" | \
  python3 stream-tts-rust/python/translation_worker_qwen_llamacpp.py
```

### Install CosyVoice
```bash
./setup_cosyvoice.sh
```

### Test Complete Pipeline (Once CosyVoice ready)
```bash
./test_ultimate_tts.sh
```

---

## Decision Points

### Q: Continue with Qwen setup or use existing system?
**Recommendation**: Complete Qwen setup (80% done). Benefits:
- Better translation quality (BLEU 33+ vs 28-30)
- Local, no API calls
- Learning for Phase 2 C++ implementation
- Only 1-2 hours more work

### Q: Which quantization for Qwen?
**Current**: Q3_K_M (3.5GB)
**Rationale**:
- Good balance of size vs quality
- Fast enough (~200ms)
- Single file (no merging needed)
- Can upgrade to Q4_K_M later if quality issues

### Q: Install CosyVoice now or optimize Qwen first?
**Recommendation**: Install CosyVoice next
- Qwen is functional, just needs polish
- CosyVoice is critical path for Ultimate system
- Can optimize Qwen in parallel
- Want end-to-end system working ASAP

---

## Issues Encountered & Solutions

### Issue 1: Model Download Stalled
**Problem**: HuggingFace CLI download hung at 2.5GB
**Solution**: Manually copied from .cache directory
**Prevention**: Use `--local-dir-use-symlinks False` flag

### Issue 2: Wrong Model Path in Worker
**Problem**: Worker looked for 72B model instead of 7B
**Root Cause**: Script was written for 72B originally
**Solution**: Updated path in translation_worker_qwen_llamacpp.py:29

### Issue 3: llama.cpp Interactive Mode Hang
**Problem**: subprocess.run() timed out waiting for input
**Root Cause**: Conversation mode enabled by default
**Solution**: Add `--single-turn --no-cnv` flags

### Issue 4: Verbose Output from llama.cpp
**Problem**: Getting model stats, prompt echoes mixed with translation
**Solution**:
- Use `--verbosity 0` flag
- Redirect stderr to /dev/null
- Parse output to extract just translation text

---

## Resources Used

### Disk Space
- llama.cpp build: ~50MB
- Qwen Q3_K_M model: 3.5GB
- Total: ~3.6GB

### Memory (During inference)
- Model: ~3.4GB (loaded to Metal GPU)
- Context: ~14MB
- Total: ~3.5GB GPU RAM

### Time Spent This Session
- Model download troubleshooting: ~30min
- llama.cpp flag testing: ~20min
- Worker script updates: ~10min
- **Total**: ~1 hour

---

## Next Session Goals

1. **5 minutes**: Fix Qwen worker flags
2. **30 minutes**: Install and test CosyVoice
3. **15 minutes**: End-to-end Ultimate pipeline test
4. **10 minutes**: Performance comparison and documentation

**Target**: Fully working Ultimate TTS system in ~1 hour

---

**Session Notes**: Made substantial progress on Qwen setup. Model is downloaded and working, just needs worker script polish. CosyVoice installation is next critical path. System is 80% complete for Phase 1.

**Copyright 2025 Andrew Yates. All rights reserved.**
