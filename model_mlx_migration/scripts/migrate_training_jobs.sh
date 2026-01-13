#!/bin/bash
# Migration script for splitting training jobs across machines
# Created: 2025-12-25
# Usage: Review and run individual sections as needed

set -e

# Configuration - UPDATE THESE
REMOTE_HOST="newmachine"  # SSH hostname or IP
REMOTE_USER="ayates"
REMOTE_DIR="~/model_mlx_migration"

# =============================================================================
# STEP 1: Check current job status before migration
# =============================================================================
check_status() {
    echo "=== Current Training Status ==="
    for log in checkpoints/ctc_*/training.log checkpoints/emotion_*/training.log checkpoints/singing_*/training.log; do
        if [ -f "$log" ]; then
            name=$(dirname "$log" | xargs basename)
            last=$(tail -1 "$log" 2>/dev/null)
            echo "$name: $last"
        fi
    done
}

# =============================================================================
# STEP 2: Stop jobs that will be migrated
# =============================================================================
stop_jobs() {
    echo "=== Stopping jobs for migration ==="

    # Find PIDs for jobs to migrate
    SPANISH_PID=$(pgrep -f "train_ctc.*spanish" || echo "")
    FRENCH_PID=$(pgrep -f "train_ctc.*french" || echo "")
    ENGLISH_PID=$(pgrep -f "train_ctc.*english_full" || echo "")
    EMOTION_PID=$(pgrep -f "emotion_unified" || echo "")
    SINGING_PID=$(pgrep -f "singing_v1" || echo "")

    echo "Spanish PID: $SPANISH_PID"
    echo "French PID: $FRENCH_PID"
    echo "English PID: $ENGLISH_PID"
    echo "Emotion PID: $EMOTION_PID"
    echo "Singing PID: $SINGING_PID"

    read -p "Kill these processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        [ -n "$SPANISH_PID" ] && kill "$SPANISH_PID" && echo "Killed Spanish"
        [ -n "$FRENCH_PID" ] && kill "$FRENCH_PID" && echo "Killed French"
        [ -n "$ENGLISH_PID" ] && kill "$ENGLISH_PID" && echo "Killed English"
        [ -n "$EMOTION_PID" ] && kill "$EMOTION_PID" && echo "Killed Emotion"
        [ -n "$SINGING_PID" ] && kill "$SINGING_PID" && echo "Killed Singing"
    fi
}

# =============================================================================
# STEP 3: Transfer checkpoints to remote machine
# =============================================================================
transfer_checkpoints() {
    echo "=== Transferring checkpoints ==="

    # Create remote directories
    ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}/checkpoints"

    # Transfer checkpoints (only latest checkpoint to save time)
    for dir in ctc_spanish ctc_french ctc_english_full emotion_unified_v2 singing_v1; do
        if [ -d "checkpoints/$dir" ]; then
            echo "Transferring $dir..."
            # Get latest checkpoint
            latest=$(ls -t checkpoints/$dir/step_*.npz 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                rsync -avz --progress "$latest" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/checkpoints/$dir/
                rsync -avz checkpoints/$dir/training.log ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/checkpoints/$dir/
            fi
        fi
    done
}

# =============================================================================
# STEP 4: Transfer datasets to remote machine
# =============================================================================
transfer_datasets() {
    echo "=== Transferring datasets ==="
    echo "This will take a while (~3 hours for all datasets)"

    # Create remote directories
    ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}/data/multilingual"

    # Spanish (28GB)
    echo "Transferring Spanish MLS (28GB)..."
    rsync -avz --progress data/multilingual/spanish/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/multilingual/spanish/

    # French (28GB)
    echo "Transferring French MLS (28GB)..."
    rsync -avz --progress data/multilingual/french/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/multilingual/french/

    # English LibriSpeech (60GB)
    echo "Transferring English LibriSpeech (60GB)..."
    rsync -avz --progress data/LibriSpeech_full/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/LibriSpeech_full/

    # Emotion data
    echo "Transferring Emotion data..."
    rsync -avz --progress data/emotion/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/emotion/

    # Singing data
    echo "Transferring Singing data..."
    rsync -avz --progress data/prosody/ravdess/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/prosody/ravdess/
    rsync -avz --progress data/singing/vocalset/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/singing/vocalset/
}

# =============================================================================
# STEP 5: Transfer code to remote machine
# =============================================================================
transfer_code() {
    echo "=== Transferring code ==="
    rsync -avz --exclude='checkpoints' --exclude='data' --exclude='.git' \
        --exclude='*.npz' --exclude='*.log' --exclude='worker_logs' \
        ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
}

# =============================================================================
# STEP 6: Generate startup commands for remote machine
# =============================================================================
generate_remote_commands() {
    echo "=== Commands to run on remote machine ==="

    # Find latest checkpoints
    SPANISH_CKPT=$(ls -t checkpoints/ctc_spanish/step_*.npz 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "step_50000.npz")
    FRENCH_CKPT=$(ls -t checkpoints/ctc_french/step_*.npz 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "step_55000.npz")
    ENGLISH_CKPT=$(ls -t checkpoints/ctc_english_full/step_*.npz 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "step_16000.npz")
    EMOTION_CKPT=$(ls -t checkpoints/emotion_unified_v2/step_*.npz 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "step_15000.npz")
    SINGING_CKPT=$(ls -t checkpoints/singing_v1/step_*.npz 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "step_13000.npz")

    cat << EOF

# Run these on the remote machine after transfer completes:

cd ${REMOTE_DIR}

# Spanish CTC (resume)
nohup python3 -m tools.whisper_mlx.train_ctc \\
    --data-dir data/multilingual/spanish/mls_spanish_opus \\
    --output-dir checkpoints/ctc_spanish \\
    --mls --epochs 3 --batch-size 4 --spec-augment --label-smoothing 0.1 \\
    --resume checkpoints/ctc_spanish/${SPANISH_CKPT} \\
    > checkpoints/ctc_spanish_training.log 2>&1 &

# French CTC (resume)
nohup python3 -m tools.whisper_mlx.train_ctc \\
    --data-dir data/multilingual/french/mls_french_opus \\
    --output-dir checkpoints/ctc_french \\
    --mls --epochs 3 --batch-size 4 --spec-augment --label-smoothing 0.1 \\
    --resume checkpoints/ctc_french/${FRENCH_CKPT} \\
    > checkpoints/ctc_french_training.log 2>&1 &

# English CTC (resume)
nohup python3 -m tools.whisper_mlx.train_ctc \\
    --data-dir data/LibriSpeech_full \\
    --output-dir checkpoints/ctc_english_full \\
    --model-size large-v3 --epochs 3 --batch-size 4 --spec-augment --label-smoothing 0.1 \\
    --resume checkpoints/ctc_english_full/${ENGLISH_CKPT} \\
    > checkpoints/ctc_english_training.log 2>&1 &

# Emotion (resume)
nohup python3 -m tools.whisper_mlx.train_multi_head \\
    --unified-emotion-dir data/emotion/unified_emotion \\
    --output-dir checkpoints/emotion_unified_v2 \\
    --train-ctc false --train-singing false --train-pitch false --train-emotion true \\
    --epochs 15 --batch-size 4 --lr 5e-5 \\
    --resume checkpoints/emotion_unified_v2/${EMOTION_CKPT} \\
    > checkpoints/emotion_training.log 2>&1 &

# Singing (resume)
nohup python3 -m tools.whisper_mlx.train_multi_head \\
    --ravdess-dir data/prosody/ravdess \\
    --vocalset-dir data/singing/vocalset/FULL \\
    --output-dir checkpoints/singing_v1 \\
    --train-ctc false --train-emotion false --train-pitch false --train-singing true \\
    --use-extended-singing --epochs 10 --batch-size 4 --lr 3e-5 \\
    --resume checkpoints/singing_v1/${SINGING_CKPT} \\
    > checkpoints/singing_training.log 2>&1 &

echo "Started 5 training jobs on remote machine"
EOF
}

# =============================================================================
# MAIN MENU
# =============================================================================
echo "Training Job Migration Script"
echo "=============================="
echo "1. Check current status"
echo "2. Stop jobs for migration"
echo "3. Transfer code only"
echo "4. Transfer checkpoints"
echo "5. Transfer datasets (slow - ~3 hrs)"
echo "6. Generate remote startup commands"
echo "7. Full migration (2+3+4+5+6)"
echo "0. Exit"
echo

read -p "Select option: " choice

case $choice in
    1) check_status ;;
    2) stop_jobs ;;
    3) transfer_code ;;
    4) transfer_checkpoints ;;
    5) transfer_datasets ;;
    6) generate_remote_commands ;;
    7)
        stop_jobs
        transfer_code
        transfer_checkpoints
        transfer_datasets
        generate_remote_commands
        ;;
    0) exit 0 ;;
    *) echo "Invalid option" ;;
esac
