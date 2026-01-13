// Copyright 2024-2025 Andrew Yates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Kokoro Prosody Parser Unit Tests
 *
 * Tests for SSML-style marker parsing and prosody adjustments.
 */

#include "prosody_parser.h"
#include "prosody_adjust.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace kokoro;

// Test helper
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Testing " #name "... "; \
    test_##name(); \
    std::cout << "PASSED\n"; \
} while(0)

// Test parsing plain text (no markers)
TEST(plain_text) {
    auto result = parse_prosody_markers("Hello world");
    assert(result.clean_text == "Hello world");
    assert(result.annotations.empty());
    assert(result.breaks.empty());
}

// Test emphasis marker
TEST(emphasis) {
    auto result = parse_prosody_markers("I <em>really</em> need this");
    assert(result.clean_text == "I really need this");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].char_start == 2);
    assert(result.annotations[0].char_end == 8);
    assert(result.annotations[0].type == ProsodyType::EMPHASIS);
}

// Test strong emphasis marker
TEST(strong_emphasis) {
    auto result = parse_prosody_markers("This is <strong>very important</strong>");
    assert(result.clean_text == "This is very important");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::STRONG_EMPHASIS);
}

// Test break with time
TEST(break_time) {
    auto result = parse_prosody_markers("Hello<break time='500ms'/>world");
    assert(result.clean_text == "Helloworld");
    assert(result.breaks.size() == 1);
    assert(result.breaks[0].after_char == 5);
    assert(result.breaks[0].duration_ms == 500);
}

// Test break with strength
TEST(break_strength) {
    auto result = parse_prosody_markers("Hello<break strength='strong'/>world");
    assert(result.clean_text == "Helloworld");
    assert(result.breaks.size() == 1);
    assert(result.breaks[0].duration_ms == 750);  // strong = 750ms
}

// Test break with seconds
TEST(break_seconds) {
    auto result = parse_prosody_markers("Hello<break time='1s'/>world");
    assert(result.clean_text == "Helloworld");
    assert(result.breaks.size() == 1);
    assert(result.breaks[0].duration_ms == 1000);
}

// Test prosody rate slow
TEST(prosody_rate_slow) {
    auto result = parse_prosody_markers("<prosody rate='slow'>Slow text</prosody>");
    assert(result.clean_text == "Slow text");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::RATE_SLOW);
}

// Test prosody rate fast
TEST(prosody_rate_fast) {
    auto result = parse_prosody_markers("<prosody rate='fast'>Fast text</prosody>");
    assert(result.clean_text == "Fast text");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::RATE_FAST);
}

// Test prosody pitch high
TEST(prosody_pitch_high) {
    auto result = parse_prosody_markers("<prosody pitch='high'>High pitch</prosody>");
    assert(result.clean_text == "High pitch");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::PITCH_HIGH);
}

// Test prosody rate with percentage value
TEST(prosody_rate_percent) {
    auto result = parse_prosody_markers("<prosody rate='80%'>Text</prosody>");
    assert(result.clean_text == "Text");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::RATE_SLOW);
    assert(std::abs(result.annotations[0].value - 0.8f) < 1e-5f);
}

// Test prosody rate with relative percentage value
TEST(prosody_rate_relative_percent) {
    auto result = parse_prosody_markers("<prosody rate='+20%'>Text</prosody>");
    assert(result.clean_text == "Text");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::RATE_FAST);
    assert(std::abs(result.annotations[0].value - 1.2f) < 1e-5f);
}

// Test prosody pitch with semitone and percent values
TEST(prosody_pitch_numeric) {
    {
        auto result = parse_prosody_markers("<prosody pitch='+2st'>Text</prosody>");
        assert(result.clean_text == "Text");
        assert(result.annotations.size() == 1);
        assert(result.annotations[0].type == ProsodyType::PITCH_HIGH);
        float expected = std::pow(2.0f, 2.0f / 12.0f);
        assert(std::abs(result.annotations[0].value - expected) < 1e-5f);
    }
    {
        auto result = parse_prosody_markers("<prosody pitch='-10%'>Text</prosody>");
        assert(result.clean_text == "Text");
        assert(result.annotations.size() == 1);
        assert(result.annotations[0].type == ProsodyType::PITCH_LOW);
        assert(std::abs(result.annotations[0].value - 0.9f) < 1e-5f);
    }
}

// Test emotion angry
TEST(emotion_angry) {
    auto result = parse_prosody_markers("<emotion type='angry'>I am angry</emotion>");
    assert(result.clean_text == "I am angry");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::EMOTION_ANGRY);
}

// Test emotion sad
TEST(emotion_sad) {
    auto result = parse_prosody_markers("<emotion type='sad'>I am sad</emotion>");
    assert(result.clean_text == "I am sad");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::EMOTION_SAD);
}

// Test emotion excited
TEST(emotion_excited) {
    auto result = parse_prosody_markers("<emotion type='excited'>So exciting</emotion>");
    assert(result.clean_text == "So exciting");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::EMOTION_EXCITED);
}

// Test whisper tag
TEST(whisper) {
    auto result = parse_prosody_markers("<whisper>Secret message</whisper>");
    assert(result.clean_text == "Secret message");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::WHISPER);
}

// Test loud tag
TEST(loud) {
    auto result = parse_prosody_markers("<loud>Loud message</loud>");
    assert(result.clean_text == "Loud message");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::LOUD);
}

// Test question tag
TEST(question) {
    auto result = parse_prosody_markers("<question>Is this a question</question>");
    assert(result.clean_text == "Is this a question");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::QUESTION);
}

// Test multiple markers
TEST(multiple_markers) {
    auto result = parse_prosody_markers(
        "I <em>really</em> need <strong>help</strong>!"
    );
    assert(result.clean_text == "I really need help!");
    assert(result.annotations.size() == 2);
    assert(result.annotations[0].type == ProsodyType::EMPHASIS);
    assert(result.annotations[1].type == ProsodyType::STRONG_EMPHASIS);
}

// Test nested markers (inner should override)
TEST(nested_markers) {
    auto result = parse_prosody_markers(
        "<emotion type='worried'>I am <em>really</em> worried</emotion>"
    );
    assert(result.clean_text == "I am really worried");
    assert(result.annotations.size() == 2);
}

// Test complex example
TEST(complex_example) {
    auto result = parse_prosody_markers(
        "I <em>really</em> understand.<break time='300ms'/> Let me help."
    );
    assert(result.clean_text == "I really understand. Let me help.");
    assert(result.annotations.size() == 1);
    assert(result.breaks.size() == 1);
    assert(result.breaks[0].duration_ms == 300);
}

// Test emphasis with level attribute
TEST(emphasis_level) {
    auto result = parse_prosody_markers(
        "<emphasis level='strong'>Very important</emphasis>"
    );
    assert(result.clean_text == "Very important");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::STRONG_EMPHASIS);
}

// Test case insensitivity
TEST(case_insensitive) {
    auto result = parse_prosody_markers("<EM>text</EM>");
    assert(result.clean_text == "text");
    assert(result.annotations.size() == 1);
    assert(result.annotations[0].type == ProsodyType::EMPHASIS);
}

// Test double quotes
TEST(double_quotes) {
    auto result = parse_prosody_markers("<break time=\"500ms\"/>");
    assert(result.breaks.size() == 1);
    assert(result.breaks[0].duration_ms == 500);
}

// Test prosody adjustment values
TEST(adjustment_values) {
    // Test EMPHASIS
    auto adj = get_adjustment(ProsodyType::EMPHASIS);
    assert(adj.duration_mult > 1.0f);  // Should be longer
    assert(adj.f0_mult > 1.0f);        // Should be higher pitch

    // Test RATE_SLOW
    adj = get_adjustment(ProsodyType::RATE_SLOW);
    assert(adj.duration_mult > 1.0f);  // Should be longer

    // Test RATE_FAST
    adj = get_adjustment(ProsodyType::RATE_FAST);
    assert(adj.duration_mult < 1.0f);  // Should be shorter

    // Test EMOTION_ANGRY
    adj = get_adjustment(ProsodyType::EMOTION_ANGRY);
    assert(adj.duration_mult < 1.0f);  // Should be faster
    assert(adj.f0_mult > 1.0f);        // Should be higher
}

// Test apply_prosody_adjustments
TEST(apply_adjustments) {
    std::vector<float> durations = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> f0 = {200.0f, 200.0f, 200.0f, 200.0f, 200.0f};

    PhonemeProsody prosody(5);
    prosody.mask[1] = ProsodyType::EMPHASIS;
    prosody.mask[2] = ProsodyType::EMPHASIS;

    apply_prosody_adjustments(durations, f0, prosody);

    // Check that emphasis was applied
    assert(durations[0] == 1.0f);  // Unchanged
    assert(durations[1] > 1.0f);   // Emphasized (longer)
    assert(durations[2] > 1.0f);   // Emphasized (longer)
    assert(durations[3] == 1.0f);  // Unchanged
}

// Test map_to_phonemes_simple
TEST(map_to_phonemes_simple_test) {
    ParsedProsody parsed;
    parsed.clean_text = "Hello world";  // 11 chars
    parsed.annotations.emplace_back(0, 5, ProsodyType::EMPHASIS);  // "Hello"

    // Map to 22 phonemes (2x ratio)
    auto prosody = map_to_phonemes_simple(parsed, 22);

    // First ~10 phonemes should be EMPHASIS
    int em_count = 0;
    for (size_t i = 0; i < prosody.mask.size(); i++) {
        if (prosody.mask[i] == ProsodyType::EMPHASIS) em_count++;
    }
    assert(em_count > 0);  // Some should be emphasized
    assert(em_count < 22); // Not all
}

// Test PhonemeProsody::has_prosody
TEST(has_prosody) {
    PhonemeProsody empty(5);
    assert(!empty.has_prosody());

    PhonemeProsody with_type(5);
    with_type.mask[2] = ProsodyType::EMPHASIS;
    assert(with_type.has_prosody());

    PhonemeProsody with_break(5);
    with_break.break_after_ms[3] = 500;
    assert(with_break.has_prosody());
}

// Test all emotion types parse correctly
TEST(all_emotions) {
    const char* emotions[] = {
        "angry", "sad", "excited", "worried", "alarmed",
        "calm", "empathetic", "confident", "frustrated",
        "nervous", "surprised", "disappointed"
    };

    for (const char* emotion : emotions) {
        std::string input = "<emotion type='" + std::string(emotion) + "'>text</emotion>";
        auto result = parse_prosody_markers(input);
        assert(result.clean_text == "text");
        assert(result.annotations.size() == 1);
        assert(result.annotations[0].type != ProsodyType::NEUTRAL);
    }
}

// Test unknown tag is ignored
TEST(unknown_tag) {
    auto result = parse_prosody_markers("<unknown>text</unknown>");
    assert(result.clean_text == "text");
    assert(result.annotations.empty());  // Unknown tag ignored
}

// Test malformed tags handled gracefully
TEST(malformed_tags) {
    // Missing closing >
    auto result1 = parse_prosody_markers("Hello <em world");
    assert(result1.clean_text.find("Hello") != std::string::npos);

    // Unclosed tag (should still capture annotation)
    auto result2 = parse_prosody_markers("Hello <em>world");
    assert(result2.clean_text == "Hello world");
}

int main() {
    std::cout << "=== Kokoro Prosody Parser Tests ===\n\n";

    // Parser tests
    RUN_TEST(plain_text);
    RUN_TEST(emphasis);
    RUN_TEST(strong_emphasis);
    RUN_TEST(break_time);
    RUN_TEST(break_strength);
    RUN_TEST(break_seconds);
    RUN_TEST(prosody_rate_slow);
    RUN_TEST(prosody_rate_fast);
    RUN_TEST(prosody_pitch_high);
    RUN_TEST(prosody_rate_percent);
    RUN_TEST(prosody_rate_relative_percent);
    RUN_TEST(prosody_pitch_numeric);
    RUN_TEST(emotion_angry);
    RUN_TEST(emotion_sad);
    RUN_TEST(emotion_excited);
    RUN_TEST(whisper);
    RUN_TEST(loud);
    RUN_TEST(question);
    RUN_TEST(multiple_markers);
    RUN_TEST(nested_markers);
    RUN_TEST(complex_example);
    RUN_TEST(emphasis_level);
    RUN_TEST(case_insensitive);
    RUN_TEST(double_quotes);
    RUN_TEST(all_emotions);
    RUN_TEST(unknown_tag);
    RUN_TEST(malformed_tags);

    // Adjustment tests
    RUN_TEST(adjustment_values);
    RUN_TEST(apply_adjustments);
    RUN_TEST(map_to_phonemes_simple_test);
    RUN_TEST(has_prosody);

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
