# Quick Start: Voice Processing for Claude Code

Get started with live TTS and multi-language translation in 3 steps.

## Step 1: Setup (2 minutes)

```bash
cd voice
./setup.sh
```

Wait for dependencies to install.

## Step 2: Test (30 seconds)

```bash
./test_tts.sh      # Japanese (default)
./test_tts.sh es   # Try Spanish
./test_tts.sh fr   # Try French
```

You should hear speech in your selected language. If you do, you're ready!

## Step 3: Use It

### Option A: One Command

```bash
# Japanese (default)
./claude_code_tts.sh "explain the code in main.py"

# Spanish
./claude_code_tts.sh --language es "explain the code in main.py"

# French
./claude_code_tts.sh --language fr "explain the code in main.py"
```

### Option B: Autonomous Worker

```bash
# Japanese (default)
./run_worker_tts.sh

# Spanish
./run_worker_tts.sh --language es
```

That's it! Claude Code will now speak to you in your chosen language.

---

## Customize

### See Available Languages

```bash
./tts_translate_stream.py --list-languages
```

Supported: English, Japanese, Spanish, French, German, Chinese, Korean, Portuguese, Italian, Russian, Arabic, Hindi

### Change Default Language

```bash
# Set environment variable
export TTS_LANGUAGE=es   # Spanish
export TTS_LANGUAGE=fr   # French
export TTS_LANGUAGE=de   # German

# Then run normally
./claude_code_tts.sh "your prompt"
```

### Adjust Speech Speed

```bash
./claude_code_tts.sh --speed "+50%" "your prompt"  # Faster
./claude_code_tts.sh --speed "-20%" "your prompt"  # Slower
```

---

## Need Help?

- Read [USAGE.md](USAGE.md) for detailed guide
- Read [README.md](README.md) for architecture
- Run `./test_tts.sh` to diagnose issues

---

## Common Issues

**No sound?**
- Check system volume
- Verify speakers/headphones connected

**Translation errors?**
- Check internet connection
- System falls back to English automatically

**Import errors?**
- Run `./setup.sh` again
- Activate venv: `source venv/bin/activate`

---

That's all you need to get started! 🎉
