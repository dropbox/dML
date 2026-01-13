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
 * Kokoro Prosody Parser Implementation
 *
 * See: prosody_parser.h for interface documentation
 */

#include "prosody_parser.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <regex>
#include <sstream>

namespace kokoro {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Trim whitespace from string
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Parse time value like "500ms" or "1s" to milliseconds
int parse_time_ms(const std::string& time_str) {
    if (time_str.empty()) return 0;

    // Try to extract number
    size_t idx = 0;
    float value = 0;
    try {
        value = std::stof(time_str, &idx);
    } catch (...) {
        return 500;  // Default
    }

    std::string unit = time_str.substr(idx);
    if (unit == "s" || unit == "sec") {
        return static_cast<int>(value * 1000);
    } else if (unit == "ms" || unit == "msec" || unit.empty()) {
        return static_cast<int>(value);
    }
    return static_cast<int>(value);
}

// Get attribute value from tag string
// e.g., get_attribute("rate='slow'", "rate") -> "slow"
std::string get_attribute(const std::string& tag_content, const std::string& attr_name) {
    // Pattern: attr_name="value" or attr_name='value' (case-insensitive)
    std::regex re(attr_name + R"(\s*=\s*(['"])(.*?)\1)", std::regex::icase);
    std::smatch m;
    if (std::regex_search(tag_content, m, re)) {
        return m[2].str();
    }

    // Fallback: unquoted value (non-SSML but convenient for casual markup)
    std::regex re_unquoted(attr_name + R"(\s*=\s*([^\s/>]+))", std::regex::icase);
    if (std::regex_search(tag_content, m, re_unquoted)) {
        return m[1].str();
    }

    return "";
}

struct ParsedAttr {
    ProsodyType type = ProsodyType::NEUTRAL;
    float value = 0.0f;  // Optional multiplier
};

bool try_parse_percent_multiplier(const std::string& s, float& out_multiplier) {
    std::string t = trim(to_lower(s));
    if (t.empty() || t.back() != '%') return false;

    std::string num = trim(t.substr(0, t.size() - 1));
    if (num.empty()) return false;

    float pct = 0.0f;
    try {
        pct = std::stof(num);
    } catch (...) {
        return false;
    }

    bool relative = (num[0] == '+' || num[0] == '-');
    if (relative) {
        out_multiplier = 1.0f + (pct / 100.0f);
    } else {
        out_multiplier = pct / 100.0f;
    }

    if (!std::isfinite(out_multiplier) || out_multiplier <= 0.0f) {
        return false;
    }
    return true;
}

bool try_parse_semitone_multiplier(const std::string& s, float& out_multiplier) {
    std::string t = trim(to_lower(s));
    if (t.size() < 3) return false;
    if (t.size() < 2 || t.substr(t.size() - 2) != "st") return false;

    std::string num = trim(t.substr(0, t.size() - 2));
    if (num.empty()) return false;

    float semitones = 0.0f;
    try {
        semitones = std::stof(num);
    } catch (...) {
        return false;
    }

    out_multiplier = std::pow(2.0f, semitones / 12.0f);
    if (!std::isfinite(out_multiplier) || out_multiplier <= 0.0f) {
        return false;
    }
    return true;
}

ParsedAttr parse_rate(const std::string& rate) {
    ParsedAttr out;
    std::string r = trim(to_lower(rate));
    if (r.empty() || r == "medium" || r == "normal" || r == "default") {
        out.type = ProsodyType::NEUTRAL;
        out.value = 1.0f;
        return out;
    }

    if (r == "x-slow" || r == "xslow") {
        out.type = ProsodyType::RATE_X_SLOW;
        out.value = 0.5f;
        return out;
    }
    if (r == "slow") {
        out.type = ProsodyType::RATE_SLOW;
        out.value = 0.7f;
        return out;
    }
    if (r == "fast") {
        out.type = ProsodyType::RATE_FAST;
        out.value = 1.3f;
        return out;
    }
    if (r == "x-fast" || r == "xfast") {
        out.type = ProsodyType::RATE_X_FAST;
        out.value = 1.6f;
        return out;
    }

    float multiplier = 0.0f;
    if (!try_parse_percent_multiplier(r, multiplier)) {
        return out;
    }

    out.value = multiplier;

    // Coarse mapping to nearest supported bucket; the parsed multiplier is preserved in `value`.
    if (multiplier <= 0.60f) return {ProsodyType::RATE_X_SLOW, multiplier};
    if (multiplier <= 0.90f) return {ProsodyType::RATE_SLOW, multiplier};
    if (multiplier >= 1.50f) return {ProsodyType::RATE_X_FAST, multiplier};
    if (multiplier >= 1.10f) return {ProsodyType::RATE_FAST, multiplier};
    return {ProsodyType::NEUTRAL, multiplier};
}

ParsedAttr parse_pitch(const std::string& pitch) {
    ParsedAttr out;
    std::string p = trim(to_lower(pitch));
    if (p.empty() || p == "medium" || p == "normal" || p == "default") {
        out.type = ProsodyType::NEUTRAL;
        out.value = 1.0f;
        return out;
    }

    if (p == "x-low" || p == "xlow") {
        out.type = ProsodyType::PITCH_X_LOW;
        out.value = 0.75f;
        return out;
    }
    if (p == "low") {
        out.type = ProsodyType::PITCH_LOW;
        out.value = 0.90f;
        return out;
    }
    if (p == "high") {
        out.type = ProsodyType::PITCH_HIGH;
        out.value = 1.15f;
        return out;
    }
    if (p == "x-high" || p == "xhigh") {
        out.type = ProsodyType::PITCH_X_HIGH;
        out.value = 1.30f;
        return out;
    }

    float multiplier = 0.0f;
    if (!try_parse_semitone_multiplier(p, multiplier) && !try_parse_percent_multiplier(p, multiplier)) {
        return out;
    }

    out.value = multiplier;

    // Coarse mapping to nearest supported bucket; the parsed multiplier is preserved in `value`.
    if (multiplier <= 0.82f) return {ProsodyType::PITCH_X_LOW, multiplier};
    if (multiplier <= 0.97f) return {ProsodyType::PITCH_LOW, multiplier};
    if (multiplier >= 1.23f) return {ProsodyType::PITCH_X_HIGH, multiplier};
    if (multiplier >= 1.06f) return {ProsodyType::PITCH_HIGH, multiplier};
    return {ProsodyType::NEUTRAL, multiplier};
}

// Map volume string to ProsodyType
ProsodyType parse_volume(const std::string& volume) {
    std::string v = trim(to_lower(volume));
    if (v == "silent") return ProsodyType::VOLUME_WHISPER;  // Treat as whisper
    if (v == "x-soft" || v == "xsoft") return ProsodyType::VOLUME_X_SOFT;
    if (v == "soft") return ProsodyType::VOLUME_SOFT;
    if (v == "loud") return ProsodyType::VOLUME_LOUD;
    if (v == "x-loud" || v == "xloud") return ProsodyType::VOLUME_X_LOUD;
    return ProsodyType::NEUTRAL;
}

// Map emotion string to ProsodyType
ProsodyType parse_emotion(const std::string& emotion) {
    std::string e = trim(to_lower(emotion));
    if (e == "angry") return ProsodyType::EMOTION_ANGRY;
    if (e == "sad") return ProsodyType::EMOTION_SAD;
    if (e == "excited") return ProsodyType::EMOTION_EXCITED;
    if (e == "worried") return ProsodyType::EMOTION_WORRIED;
    if (e == "alarmed") return ProsodyType::EMOTION_ALARMED;
    if (e == "calm") return ProsodyType::EMOTION_CALM;
    if (e == "empathetic") return ProsodyType::EMOTION_EMPATHETIC;
    if (e == "confident") return ProsodyType::EMOTION_CONFIDENT;
    if (e == "frustrated") return ProsodyType::EMOTION_FRUSTRATED;
    if (e == "nervous") return ProsodyType::EMOTION_NERVOUS;
    if (e == "surprised") return ProsodyType::EMOTION_SURPRISED;
    if (e == "disappointed") return ProsodyType::EMOTION_DISAPPOINTED;
    return ProsodyType::NEUTRAL;
}

// Map emphasis level to ProsodyType
ProsodyType parse_emphasis_level(const std::string& level) {
    if (level == "strong") return ProsodyType::STRONG_EMPHASIS;
    if (level == "moderate" || level == "normal" || level.empty()) return ProsodyType::EMPHASIS;
    if (level == "reduced" || level == "none") return ProsodyType::REDUCED_EMPHASIS;
    return ProsodyType::EMPHASIS;
}

// Structure to track open tags
struct TagState {
    std::string tag_name;
    ProsodyType type;
    size_t clean_start;  // Start position in clean text
    float value;
};

}  // namespace

ParsedProsody parse_prosody_markers(const std::string& text) {
    ParsedProsody result;
    std::vector<TagState> tag_stack;

    size_t i = 0;
    size_t text_len = text.length();

    while (i < text_len) {
        // Check for tag start
        if (text[i] == '<') {
            // Find tag end
            size_t tag_end = text.find('>', i);
            if (tag_end == std::string::npos) {
                // No closing >, treat as text
                result.clean_text += text[i];
                i++;
                continue;
            }

            std::string tag_content = text.substr(i + 1, tag_end - i - 1);

            // Check if self-closing tag (ends with /)
            bool self_closing = !tag_content.empty() && tag_content.back() == '/';
            if (self_closing) {
                tag_content = tag_content.substr(0, tag_content.length() - 1);
            }

            tag_content = trim(tag_content);

            // Check if closing tag
            bool closing = !tag_content.empty() && tag_content[0] == '/';
            if (closing) {
                tag_content = tag_content.substr(1);
            }

            // Extract tag name (first word)
            std::string tag_name;
            size_t space_pos = tag_content.find_first_of(" \t\n");
            if (space_pos != std::string::npos) {
                tag_name = tag_content.substr(0, space_pos);
            } else {
                tag_name = tag_content;
            }

            // Convert to lowercase for case-insensitive matching
            std::transform(tag_name.begin(), tag_name.end(), tag_name.begin(), ::tolower);

            // Handle closing tags
            if (closing) {
                // Find matching open tag (search from end)
                for (auto it = tag_stack.rbegin(); it != tag_stack.rend(); ++it) {
                    if (it->tag_name == tag_name) {
                        // Create annotation for this span
                        result.annotations.emplace_back(
                            it->clean_start,
                            result.clean_text.length(),
                            it->type,
                            it->value
                        );
                        // Remove from stack
                        tag_stack.erase(std::next(it).base());
                        break;
                    }
                }
                i = tag_end + 1;
                continue;
            }

            // Handle self-closing tags
            if (self_closing || tag_name == "break" || tag_name == "br") {
                // Break tag
                if (tag_name == "break" || tag_name == "br") {
                    std::string time_attr = get_attribute(tag_content, "time");
                    std::string strength_attr = get_attribute(tag_content, "strength");

                    int duration_ms;
                    if (!time_attr.empty()) {
                        duration_ms = parse_time_ms(time_attr);
                    } else if (!strength_attr.empty()) {
                        duration_ms = break_strength_to_ms(strength_attr.c_str());
                    } else {
                        duration_ms = 500;  // Default
                    }

                    if (duration_ms > 0) {
                        result.breaks.emplace_back(result.clean_text.length(), duration_ms);
                    }
                }
                i = tag_end + 1;
                continue;
            }

            // Handle opening tags
            TagState state;
            state.tag_name = tag_name;
            state.clean_start = result.clean_text.length();
            state.value = 0.0f;
            state.type = ProsodyType::NEUTRAL;

            if (tag_name == "em" || tag_name == "emphasis") {
                std::string level = get_attribute(tag_content, "level");
                state.type = parse_emphasis_level(level);
            } else if (tag_name == "strong") {
                state.type = ProsodyType::STRONG_EMPHASIS;
            } else if (tag_name == "prosody") {
                // Check for rate, pitch, volume attributes
                std::string rate = get_attribute(tag_content, "rate");
                std::string pitch = get_attribute(tag_content, "pitch");
                std::string volume = get_attribute(tag_content, "volume");

                if (!rate.empty()) {
                    auto parsed = parse_rate(rate);
                    state.type = parsed.type;
                    state.value = parsed.value;
                } else if (!pitch.empty()) {
                    auto parsed = parse_pitch(pitch);
                    state.type = parsed.type;
                    state.value = parsed.value;
                } else if (!volume.empty()) {
                    state.type = parse_volume(volume);
                }
            } else if (tag_name == "emotion") {
                std::string type = get_attribute(tag_content, "type");
                state.type = parse_emotion(type);
            } else if (tag_name == "whisper") {
                state.type = ProsodyType::WHISPER;
            } else if (tag_name == "loud") {
                state.type = ProsodyType::LOUD;
            } else if (tag_name == "question") {
                state.type = ProsodyType::QUESTION;
            }

            // Only push if we recognized the tag
            if (state.type != ProsodyType::NEUTRAL) {
                tag_stack.push_back(state);
            }

            i = tag_end + 1;
        } else {
            // Regular character
            result.clean_text += text[i];
            i++;
        }
    }

    // Close any remaining open tags
    for (auto it = tag_stack.rbegin(); it != tag_stack.rend(); ++it) {
        result.annotations.emplace_back(
            it->clean_start,
            result.clean_text.length(),
            it->type,
            it->value
        );
    }

    return result;
}

PhonemeProsody map_to_phonemes(
    const ParsedProsody& parsed,
    const std::string& phonemes,
    const std::vector<int>& char_to_phoneme
) {
    size_t num_phonemes = phonemes.length();
    PhonemeProsody result(num_phonemes);

    // Map annotations to phonemes
    for (const auto& ann : parsed.annotations) {
        // Find phoneme range for this character span
        int ph_start = -1;
        int ph_end = -1;

        for (size_t c = ann.char_start; c < ann.char_end && c < char_to_phoneme.size(); c++) {
            int ph = char_to_phoneme[c];
            if (ph >= 0 && ph < static_cast<int>(num_phonemes)) {
                if (ph_start < 0 || ph < ph_start) ph_start = ph;
                if (ph > ph_end) ph_end = ph;
            }
        }

        // Apply prosody type to phoneme range
        if (ph_start >= 0 && ph_end >= 0) {
            for (int p = ph_start; p <= ph_end; p++) {
                // Only override if current is NEUTRAL or new type has higher priority
                if (result.mask[p] == ProsodyType::NEUTRAL ||
                    static_cast<int>(ann.type) > static_cast<int>(result.mask[p])) {
                    result.mask[p] = ann.type;
                }
            }
        }
    }

    // Map breaks to phonemes
    for (const auto& brk : parsed.breaks) {
        // Find phoneme after this character position
        if (brk.after_char < char_to_phoneme.size()) {
            int ph = char_to_phoneme[brk.after_char];
            // Look backwards for valid phoneme index
            for (size_t c = brk.after_char; c > 0 && ph < 0; c--) {
                ph = char_to_phoneme[c - 1];
            }
            if (ph >= 0 && ph < static_cast<int>(num_phonemes)) {
                result.break_after_ms[ph] = brk.duration_ms;
            }
        }
    }

    return result;
}

PhonemeProsody map_to_phonemes_simple(
    const ParsedProsody& parsed,
    size_t num_phonemes
) {
    PhonemeProsody result(num_phonemes);

    if (parsed.clean_text.empty() || num_phonemes == 0) {
        return result;
    }

    // Simple heuristic: linear mapping from characters to phonemes
    float ratio = static_cast<float>(num_phonemes) / parsed.clean_text.length();

    // Map annotations
    for (const auto& ann : parsed.annotations) {
        size_t ph_start = static_cast<size_t>(ann.char_start * ratio);
        size_t ph_end = static_cast<size_t>(ann.char_end * ratio);

        // Clamp to valid range
        ph_start = std::min(ph_start, num_phonemes - 1);
        ph_end = std::min(ph_end, num_phonemes);

        for (size_t p = ph_start; p < ph_end; p++) {
            if (result.mask[p] == ProsodyType::NEUTRAL ||
                static_cast<int>(ann.type) > static_cast<int>(result.mask[p])) {
                result.mask[p] = ann.type;
            }
        }
    }

    // Map breaks
    for (const auto& brk : parsed.breaks) {
        size_t ph = static_cast<size_t>(brk.after_char * ratio);
        if (ph >= num_phonemes) ph = num_phonemes - 1;
        result.break_after_ms[ph] = brk.duration_ms;
    }

    return result;
}

}  // namespace kokoro
