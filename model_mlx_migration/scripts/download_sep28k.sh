#!/bin/bash
# Download SEP-28k (Stuttering Events in Podcasts) dataset
# Reference: https://github.com/apple/ml-stuttering-events-dataset
#
# This dataset contains ~28,000 3-second audio clips labeled with:
# - Prolongation (elongated syllables)
# - Block (gasps/stuttered pauses)
# - Sound Repetition (repeated syllables)
# - Word Repetition (repeated words)
# - Interjection (um, uh)

set -e

DATA_DIR="data/sep28k"
mkdir -p "$DATA_DIR"

echo "========================================"
echo "SEP-28k Dataset Download"
echo "========================================"
echo ""
echo "The SEP-28k dataset requires manual download due to licensing."
echo ""
echo "Steps to download:"
echo "1. Clone the repository:"
echo "   git clone https://github.com/apple/ml-stuttering-events-dataset.git $DATA_DIR/repo"
echo ""
echo "2. Follow their download instructions to get the audio clips"
echo "   (requires running their Python scripts to download from podcasts)"
echo ""
echo "3. The expected directory structure:"
echo "   $DATA_DIR/"
echo "   ├── clips/"
echo "   │   ├── <ShowName>/"
echo "   │   │   ├── <EpId>/"
echo "   │   │   │   ├── <ClipId>.wav"
echo "   │   ├── SEP-28k_labels.csv"
echo "   │   └── FluencyBank_labels.csv"
echo ""
echo "Once downloaded, the train_paralinguistics.py script will automatically"
echo "load the disfluency samples using the SEP28kDataset class."
echo ""
echo "========================================"

# Check if repo exists
if [ -d "$DATA_DIR/repo" ]; then
    echo "Repository already cloned at $DATA_DIR/repo"
else
    echo "Cloning repository..."
    git clone https://github.com/apple/ml-stuttering-events-dataset.git "$DATA_DIR/repo" || {
        echo "Failed to clone repository. Please clone manually:"
        echo "git clone https://github.com/apple/ml-stuttering-events-dataset.git $DATA_DIR/repo"
    }
fi

echo ""
echo "Next steps:"
echo "1. cd $DATA_DIR/repo"
echo "2. Follow the README to download audio clips"
echo "3. Move/symlink clips to $DATA_DIR/clips/"
echo ""
echo "The training script will automatically detect and load SEP-28k data."
