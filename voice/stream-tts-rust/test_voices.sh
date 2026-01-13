#!/bin/bash
# Test different Japanese voices at 280 WPM
# Copyright 2025 Andrew Yates. All rights reserved.

echo "Testing Japanese voices at 280 WPM..."
echo ""

echo "1. Kyoko (Default female voice)"
say -v Kyoko -r 280 "こんにちは、私は京子です。自然な日本語の発音をテストしています。"
sleep 1

echo ""
echo "2. Flo (Premium female voice)"
say -v Flo -r 280 "こんにちは、私はFloです。自然な日本語の発音をテストしています。"
sleep 1

echo ""
echo "3. Eddy (Premium male voice)"
say -v Eddy -r 280 "こんにちは、私はEddyです。自然な日本語の発音をテストしています。"
sleep 1

echo ""
echo "4. Shelley (Premium female voice)"
say -v Shelley -r 280 "こんにちは、私はShelleyです。自然な日本語の発音をテストしています。"
sleep 1

echo ""
echo "5. Reed (Premium male voice)"
say -v Reed -r 280 "こんにちは、私はReedです。自然な日本語の発音をテストしています。"
sleep 1

echo ""
echo "Testing complete! Which voice did you prefer?"
