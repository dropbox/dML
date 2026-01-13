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

// Output Writers for Whisper Transcription Results
// Implementation matching Python mlx_whisper/writers.py exactly

#include "output_writers.h"
#include "mlx_inference_engine.hpp"
#include <sstream>
#include <iomanip>
#include <cassert>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace mlx_inference {

// ============================================================================
// GAP 74: format_timestamp - Exact Python implementation
// ============================================================================

std::string format_timestamp(float seconds, bool always_include_hours,
                             const std::string& decimal_marker) {
    assert(seconds >= 0 && "non-negative timestamp expected");

    // Python: milliseconds = round(seconds * 1000.0)
    int64_t milliseconds = static_cast<int64_t>(std::round(seconds * 1000.0));

    // Python: hours = milliseconds // 3_600_000
    int64_t hours = milliseconds / 3600000;
    milliseconds -= hours * 3600000;

    // Python: minutes = milliseconds // 60_000
    int64_t minutes = milliseconds / 60000;
    milliseconds -= minutes * 60000;

    // Python: seconds = milliseconds // 1_000
    int64_t secs = milliseconds / 1000;
    milliseconds -= secs * 1000;

    // Python: hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    std::ostringstream oss;
    if (always_include_hours || hours > 0) {
        oss << std::setfill('0') << std::setw(2) << hours << ":";
    }

    // Python: return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    oss << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << secs << decimal_marker
        << std::setfill('0') << std::setw(3) << milliseconds;

    return oss.str();
}

// ============================================================================
// ResultWriter base class
// ============================================================================

void ResultWriter::operator()(const TranscriptionResult& result, const std::string& output_name,
                               const SubtitleOptions& options) {
    std::filesystem::path output_path = std::filesystem::path(output_dir_) / output_name;
    output_path.replace_extension(extension());

    std::ofstream file(output_path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path.string());
    }

    write_result(result, file, options);
}

// ============================================================================
// GAP 101: WriteTXT - Plain text output
// ============================================================================

void WriteTXT::write_result(const TranscriptionResult& result, std::ostream& file,
                            const SubtitleOptions& options) {
    // Python: for segment in result["segments"]: print(segment["text"].strip())
    for (const auto& segment : result.segments) {
        // Strip leading/trailing whitespace
        std::string text = segment.text;
        size_t start = text.find_first_not_of(" \t\n\r");
        size_t end = text.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            text = text.substr(start, end - start + 1);
        } else {
            text.clear();
        }
        file << text << "\n";
        file.flush();
    }
}

// ============================================================================
// SubtitlesWriter - Base class for VTT/SRT
// ============================================================================

namespace {

// Helper: Get first word start time from segments
float get_start(const std::vector<TranscriptionSegment>& segments) {
    // Python: return next((w["start"] for s in segments for w in s["words"]), segments[0]["start"])
    for (const auto& seg : segments) {
        for (const auto& word : seg.words) {
            return word.start_time;
        }
    }
    if (!segments.empty()) {
        return segments[0].start_time;
    }
    return 0.0f;
}

// Helper: Strip leading/trailing whitespace
std::string strip(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// Helper: Replace substring
std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

} // anonymous namespace

std::vector<SubtitleEntry> SubtitlesWriter::iterate_result(const TranscriptionResult& result,
                                                            const SubtitleOptions& options) {
    std::vector<SubtitleEntry> entries;

    // Get options with defaults (Python: max_line_width or 1000, etc.)
    int max_line_width = options.max_line_width.value_or(1000);
    int max_line_count = options.max_line_count.value_or(0);  // 0 means unlimited
    int max_words_per_line = options.max_words_per_line.value_or(1000);
    bool highlight_words = options.highlight_words;
    bool preserve_segments = !options.max_line_count.has_value() || !options.max_line_width.has_value();

    // Check if we have word-level timestamps
    bool has_words = !result.segments.empty() && !result.segments[0].words.empty();

    if (has_words) {
        // Word-level iteration with line breaking (Python iterate_subtitles)
        int line_len = 0;
        int line_count = 1;
        std::vector<std::pair<WordInfo, std::string>> subtitle;  // word + display text
        float last = get_start(result.segments);

        for (const auto& segment : result.segments) {
            size_t chunk_index = 0;
            while (chunk_index < segment.words.size()) {
                size_t remaining = segment.words.size() - chunk_index;
                size_t words_count = std::min(static_cast<size_t>(max_words_per_line), remaining);

                for (size_t i = 0; i < words_count; ++i) {
                    WordInfo word = segment.words[chunk_index + i];
                    std::string display_word = word.word;

                    // GAP 108: Long pause detection (>3 seconds)
                    bool long_pause = !preserve_segments && (word.start_time - last > 3.0f);
                    bool has_room = line_len + static_cast<int>(word.word.length()) <= max_line_width;
                    bool seg_break = (i == 0) && !subtitle.empty() && preserve_segments;

                    if (line_len > 0 && has_room && !long_pause && !seg_break) {
                        // Line continuation
                        line_len += static_cast<int>(word.word.length());
                    } else {
                        // New line
                        display_word = strip(display_word);
                        if (!subtitle.empty() && max_line_count > 0 &&
                            (long_pause || line_count >= max_line_count || seg_break)) {
                            // Subtitle break - yield current subtitle
                            if (!highlight_words) {
                                std::ostringstream text;
                                for (const auto& [w, d] : subtitle) {
                                    text << d;
                                }
                                SubtitleEntry entry;
                                entry.start = format_ts(subtitle.front().first.start_time);
                                entry.end = format_ts(subtitle.back().first.end_time);
                                entry.text = text.str();
                                entries.push_back(entry);
                            } else {
                                // GAP 105: Word highlighting
                                std::string subtitle_start = format_ts(subtitle.front().first.start_time);
                                std::string subtitle_end = format_ts(subtitle.back().first.end_time);
                                std::string full_text;
                                for (const auto& [w, d] : subtitle) {
                                    full_text += d;
                                }

                                std::string last_ts = subtitle_start;
                                for (size_t j = 0; j < subtitle.size(); ++j) {
                                    std::string start_ts = format_ts(subtitle[j].first.start_time);
                                    std::string end_ts = format_ts(subtitle[j].first.end_time);

                                    if (last_ts != start_ts) {
                                        SubtitleEntry gap_entry;
                                        gap_entry.start = last_ts;
                                        gap_entry.end = start_ts;
                                        gap_entry.text = full_text;
                                        entries.push_back(gap_entry);
                                    }

                                    // Highlight current word with <u> tags
                                    std::ostringstream highlighted;
                                    for (size_t k = 0; k < subtitle.size(); ++k) {
                                        if (k == j) {
                                            // Apply <u> to word, preserving leading whitespace
                                            std::string w = subtitle[k].second;
                                            size_t first_char = w.find_first_not_of(" \t\n");
                                            if (first_char != std::string::npos && first_char > 0) {
                                                highlighted << w.substr(0, first_char)
                                                           << "<u>" << w.substr(first_char) << "</u>";
                                            } else {
                                                highlighted << "<u>" << w << "</u>";
                                            }
                                        } else {
                                            highlighted << subtitle[k].second;
                                        }
                                    }

                                    SubtitleEntry word_entry;
                                    word_entry.start = start_ts;
                                    word_entry.end = end_ts;
                                    word_entry.text = highlighted.str();
                                    entries.push_back(word_entry);

                                    last_ts = end_ts;
                                }
                            }
                            subtitle.clear();
                            line_count = 1;
                        } else if (line_len > 0) {
                            // Line break within subtitle
                            line_count++;
                            display_word = "\n" + display_word;
                        }
                        line_len = static_cast<int>(strip(display_word).length());
                    }
                    subtitle.push_back({word, display_word});
                    last = word.start_time;
                }
                chunk_index += max_words_per_line;
            }
        }

        // Yield remaining subtitle
        if (!subtitle.empty()) {
            if (!highlight_words) {
                std::ostringstream text;
                for (const auto& [w, d] : subtitle) {
                    text << d;
                }
                SubtitleEntry entry;
                entry.start = format_ts(subtitle.front().first.start_time);
                entry.end = format_ts(subtitle.back().first.end_time);
                entry.text = text.str();
                entries.push_back(entry);
            } else {
                // GAP 105: Word highlighting for final subtitle
                std::string subtitle_start = format_ts(subtitle.front().first.start_time);
                std::string subtitle_end = format_ts(subtitle.back().first.end_time);
                std::string full_text;
                for (const auto& [w, d] : subtitle) {
                    full_text += d;
                }

                std::string last_ts = subtitle_start;
                for (size_t j = 0; j < subtitle.size(); ++j) {
                    std::string start_ts = format_ts(subtitle[j].first.start_time);
                    std::string end_ts = format_ts(subtitle[j].first.end_time);

                    if (last_ts != start_ts) {
                        SubtitleEntry gap_entry;
                        gap_entry.start = last_ts;
                        gap_entry.end = start_ts;
                        gap_entry.text = full_text;
                        entries.push_back(gap_entry);
                    }

                    std::ostringstream highlighted;
                    for (size_t k = 0; k < subtitle.size(); ++k) {
                        if (k == j) {
                            std::string w = subtitle[k].second;
                            size_t first_char = w.find_first_not_of(" \t\n");
                            if (first_char != std::string::npos && first_char > 0) {
                                highlighted << w.substr(0, first_char)
                                           << "<u>" << w.substr(first_char) << "</u>";
                            } else {
                                highlighted << "<u>" << w << "</u>";
                            }
                        } else {
                            highlighted << subtitle[k].second;
                        }
                    }

                    SubtitleEntry word_entry;
                    word_entry.start = start_ts;
                    word_entry.end = end_ts;
                    word_entry.text = highlighted.str();
                    entries.push_back(word_entry);

                    last_ts = end_ts;
                }
            }
        }
    } else {
        // Segment-level iteration (no word timestamps)
        for (const auto& segment : result.segments) {
            SubtitleEntry entry;
            entry.start = format_ts(segment.start_time);
            entry.end = format_ts(segment.end_time);
            // Python: segment["text"].strip().replace("-->", "->")
            entry.text = replace_all(strip(segment.text), "-->", "->");
            entries.push_back(entry);
        }
    }

    return entries;
}

// ============================================================================
// GAP 102: WriteVTT - WebVTT with header
// ============================================================================

void WriteVTT::write_result(const TranscriptionResult& result, std::ostream& file,
                            const SubtitleOptions& options) {
    // Python: print("WEBVTT\n", file=file)
    file << "WEBVTT\n\n";

    // Python: for start, end, text in self.iterate_result(...):
    //             print(f"{start} --> {end}\n{text}\n", file=file)
    for (const auto& entry : iterate_result(result, options)) {
        file << entry.start << " --> " << entry.end << "\n"
             << entry.text << "\n\n";
        file.flush();
    }
}

// ============================================================================
// GAP 103: WriteSRT - SRT with decimal comma
// ============================================================================

void WriteSRT::write_result(const TranscriptionResult& result, std::ostream& file,
                            const SubtitleOptions& options) {
    // Python: for i, (start, end, text) in enumerate(iterate_result(...), start=1):
    //             print(f"{i}\n{start} --> {end}\n{text}\n", file=file)
    int i = 1;
    for (const auto& entry : iterate_result(result, options)) {
        file << i << "\n"
             << entry.start << " --> " << entry.end << "\n"
             << entry.text << "\n\n";
        file.flush();
        i++;
    }
}

// ============================================================================
// GAP 104: WriteTSV - Tab-separated with integer milliseconds
// ============================================================================

void WriteTSV::write_result(const TranscriptionResult& result, std::ostream& file,
                            const SubtitleOptions& options) {
    // Python: print("start", "end", "text", sep="\t", file=file)
    file << "start\tend\ttext\n";

    // Python: for segment in result["segments"]:
    //             print(round(1000 * segment["start"]), end="\t")
    //             print(round(1000 * segment["end"]), end="\t")
    //             print(segment["text"].strip().replace("\t", " "))
    for (const auto& segment : result.segments) {
        int64_t start_ms = static_cast<int64_t>(std::round(segment.start_time * 1000.0f));
        int64_t end_ms = static_cast<int64_t>(std::round(segment.end_time * 1000.0f));
        std::string text = replace_all(strip(segment.text), "\t", " ");
        file << start_ms << "\t" << end_ms << "\t" << text << "\n";
        file.flush();
    }
}

// ============================================================================
// GAP 101: WriteJSON - Full result as JSON
// ============================================================================

namespace {

// Simple JSON escaping for strings
std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

} // anonymous namespace

void WriteJSON::write_result(const TranscriptionResult& result, std::ostream& file,
                             const SubtitleOptions& options) {
    // Build JSON manually to match Python json.dump output
    file << "{";
    file << "\"text\":\"" << json_escape(result.text) << "\",";
    file << "\"language\":\"" << json_escape(result.language) << "\",";
    file << "\"segments\":[";

    for (size_t i = 0; i < result.segments.size(); ++i) {
        const auto& seg = result.segments[i];
        if (i > 0) file << ",";
        file << "{";
        file << "\"start\":" << std::fixed << std::setprecision(3) << seg.start_time << ",";
        file << "\"end\":" << std::fixed << std::setprecision(3) << seg.end_time << ",";
        file << "\"text\":\"" << json_escape(seg.text) << "\",";
        file << "\"avg_logprob\":" << std::fixed << std::setprecision(6) << seg.avg_logprob << ",";
        file << "\"no_speech_prob\":" << std::fixed << std::setprecision(6) << seg.no_speech_prob << ",";

        // Tokens array
        file << "\"tokens\":[";
        for (size_t j = 0; j < seg.tokens.size(); ++j) {
            if (j > 0) file << ",";
            file << seg.tokens[j];
        }
        file << "],";

        // Words array (if present)
        file << "\"words\":[";
        for (size_t j = 0; j < seg.words.size(); ++j) {
            const auto& word = seg.words[j];
            if (j > 0) file << ",";
            file << "{";
            file << "\"word\":\"" << json_escape(word.word) << "\",";
            file << "\"start\":" << std::fixed << std::setprecision(3) << word.start_time << ",";
            file << "\"end\":" << std::fixed << std::setprecision(3) << word.end_time << ",";
            file << "\"probability\":" << std::fixed << std::setprecision(6) << word.probability;
            file << "}";
        }
        file << "]";

        file << "}";
    }

    file << "]}";
}

// ============================================================================
// GAP 107: get_writer and write_all_formats
// ============================================================================

std::unique_ptr<ResultWriter> get_writer(const std::string& output_format,
                                          const std::string& output_dir) {
    if (output_format == "txt") {
        return std::make_unique<WriteTXT>(output_dir);
    } else if (output_format == "vtt") {
        return std::make_unique<WriteVTT>(output_dir);
    } else if (output_format == "srt") {
        return std::make_unique<WriteSRT>(output_dir);
    } else if (output_format == "tsv") {
        return std::make_unique<WriteTSV>(output_dir);
    } else if (output_format == "json") {
        return std::make_unique<WriteJSON>(output_dir);
    }
    return nullptr;
}

void write_all_formats(const TranscriptionResult& result, const std::string& output_dir,
                       const std::string& output_name, const SubtitleOptions& options) {
    // Python: all_writers = [writer(output_dir) for writer in writers.values()]
    WriteTXT txt_writer{output_dir};
    txt_writer(result, output_name, options);

    WriteVTT vtt_writer{output_dir};
    vtt_writer(result, output_name, options);

    WriteSRT srt_writer{output_dir};
    srt_writer(result, output_name, options);

    WriteTSV tsv_writer{output_dir};
    tsv_writer(result, output_name, options);

    WriteJSON json_writer{output_dir};
    json_writer(result, output_name, options);
}

std::string write_to_string(const TranscriptionResult& result, const std::string& format,
                            const SubtitleOptions& options) {
    std::ostringstream oss;

    if (format == "txt") {
        WriteTXT("").write_result(result, oss, options);
    } else if (format == "vtt") {
        WriteVTT("").write_result(result, oss, options);
    } else if (format == "srt") {
        WriteSRT("").write_result(result, oss, options);
    } else if (format == "tsv") {
        WriteTSV("").write_result(result, oss, options);
    } else if (format == "json") {
        WriteJSON("").write_result(result, oss, options);
    } else {
        throw std::invalid_argument("Unknown output format: " + format);
    }

    return oss.str();
}

} // namespace mlx_inference
