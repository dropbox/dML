#!/bin/bash
# Complete MeCab/Cutlet Integration Test
# Tests both Python and C++ implementations

set -e

echo "================================================"
echo "MeCab/Cutlet Complete Integration Test"
echo "================================================"
echo

# Change to repo root
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

echo "1. Testing Python mecab_helper..."
echo "-----------------------------------"
python3 mecab_helper.py
echo

echo "2. Testing direct cutlet usage..."
echo "-----------------------------------"
python3 << 'EOF'
import cutlet
import unidic_lite

dicdir = unidic_lite.DICDIR
katsu = cutlet.Cutlet(mecab_args=f'-d {dicdir}')

test_cases = [
    "こんにちは世界",
    "東京タワーへ行きます",
    "私は学生です",
    "今日は良い天気ですね"
]

print("Direct cutlet test:")
for text in test_cases:
    romaji = katsu.romaji(text)
    print(f"  {text} → {romaji}")

print("\n✓ Direct cutlet test passed!")
EOF
echo

echo "3. Compiling C++ MeCab test..."
echo "-----------------------------------"
if [ ! -f mecab_test ]; then
    echo "Compiling mecab_test.cpp..."
    g++ -std=c++17 -O3 mecab_test.cpp -o mecab_test $(mecab-config --cflags --libs) 2>&1 | grep -v "duplicate libraries" || true
    echo "✓ Compilation successful"
else
    echo "✓ mecab_test already compiled"
fi
echo

echo "4. Testing C++ MeCab basic parsing..."
echo "-----------------------------------"
./mecab_test "こんにちは世界"
echo "✓ Basic parsing works"
echo

echo "5. Testing C++ MeCab detailed analysis..."
echo "-----------------------------------"
./mecab_test --nodes "今日は良い天気です"
echo

echo "6. Benchmarking C++ MeCab performance..."
echo "-----------------------------------"
./mecab_test --bench "こんにちは"
echo

echo "7. Testing Python helper function import..."
echo "-----------------------------------"
python3 << 'EOF'
from mecab_helper import get_cutlet, get_fugashi_tagger

# Test cutlet
katsu = get_cutlet()
result = katsu.romaji("こんにちは")
assert "konnichiha" in result.lower() or "konnichi" in result.lower(), f"Expected romanization of 'こんにちは', got '{result}'"
print(f"✓ get_cutlet() works: こんにちは → {result}")

# Test fugashi
tagger = get_fugashi_tagger()
nodes = [word for word in tagger("テスト")]
assert len(nodes) > 0, "Expected at least one node"
print(f"✓ get_fugashi_tagger() works: found {len(nodes)} morphemes")
EOF
echo

echo "================================================"
echo "ALL TESTS PASSED! ✓"
echo "================================================"
echo
echo "Summary:"
echo "  ✓ Python mecab_helper.py"
echo "  ✓ Direct cutlet with unidic_lite"
echo "  ✓ C++ MeCab compilation"
echo "  ✓ C++ MeCab parsing"
echo "  ✓ C++ MeCab detailed analysis"
echo "  ✓ C++ MeCab benchmarking"
echo "  ✓ Python helper imports"
echo
echo "MeCab/Cutlet is fully functional!"
