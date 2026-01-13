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
// Matches Python mlx_whisper/writers.py format exactly
//
// GAPs addressed:
// - GAP 74: _format_timestamp function
// - GAP 101: Output Writers (TXT, VTT, SRT, TSV, JSON)
// - GAP 102: VTT format with WEBVTT header
// - GAP 103: SRT format with decimal comma
// - GAP 104: TSV with integer milliseconds
// - GAP 105: Word highlighting
// - GAP 106: max_line_width and max_line_count
// - GAP 107: "all" output format
// - GAP 108: long_pause detection in subtitles

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <optional>
#include <functional>
#include <regex>

namespace mlx_inference {

// Forward declarations
struct TranscriptionResult;
struct TranscriptionSegment;
struct WordInfo;

/**
 * GAP 74: Format timestamp exactly as Python does.
 *
 * Python implementation:
 *   milliseconds = round(seconds * 1000.0)
 *   hours = milliseconds // 3_600_000
 *   ...
 *   return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
 *
 * @param seconds Time in seconds (must be >= 0)
 * @param always_include_hours Include hours even when zero
 * @param decimal_marker "." for VTT, "," for SRT
 * @return Formatted timestamp string
 */
std::string format_timestamp(float seconds, bool always_include_hours = false,
                             const std::string& decimal_marker = ".");

/**
 * Options for subtitle generation.
 * Matches Python SubtitlesWriter options.
 */
struct SubtitleOptions {
    std::optional<int> max_line_width;     // GAP 106: Maximum characters per line
    std::optional<int> max_line_count;     // GAP 106: Maximum lines per subtitle
    bool highlight_words = false;          // GAP 105: Underline current word
    std::optional<int> max_words_per_line; // GAP 106: Maximum words per line
};

/**
 * A subtitle entry with timing.
 * Intermediate format for subtitle iteration.
 */
struct SubtitleEntry {
    std::string start;  // Formatted start timestamp
    std::string end;    // Formatted end timestamp
    std::string text;   // Subtitle text (may contain newlines)
};

/**
 * Base class for result writers.
 * Matches Python ResultWriter pattern.
 */
class ResultWriter {
public:
    virtual ~ResultWriter() = default;

    ResultWriter(const std::string& output_dir) : output_dir_(output_dir) {}

    /**
     * Write result to file with appropriate extension.
     * @param result Transcription result
     * @param output_name Base filename (without extension)
     * @param options Optional subtitle options
     */
    void operator()(const TranscriptionResult& result, const std::string& output_name,
                    const SubtitleOptions& options = SubtitleOptions());

    /**
     * Write result to stream.
     * Must be implemented by subclasses.
     */
    virtual void write_result(const TranscriptionResult& result, std::ostream& file,
                              const SubtitleOptions& options = SubtitleOptions()) = 0;

    /**
     * Get file extension for this writer.
     */
    virtual std::string extension() const = 0;

protected:
    std::string output_dir_;
};

/**
 * GAP 101: Plain text output.
 * Writes segment text without timestamps.
 */
class WriteTXT : public ResultWriter {
public:
    using ResultWriter::ResultWriter;

    void write_result(const TranscriptionResult& result, std::ostream& file,
                      const SubtitleOptions& options = SubtitleOptions()) override;

    std::string extension() const override { return "txt"; }
};

/**
 * Base class for subtitle formats (VTT, SRT).
 * Handles word-level iteration with line breaking and highlighting.
 */
class SubtitlesWriter : public ResultWriter {
public:
    using ResultWriter::ResultWriter;

    /**
     * Iterate over result producing subtitle entries.
     * Handles word-level timestamps, line breaking, and highlighting.
     *
     * GAP 105: Word highlighting with <u> tags
     * GAP 106: Line width/count limits
     * GAP 108: Long pause (>3s) detection
     */
    std::vector<SubtitleEntry> iterate_result(const TranscriptionResult& result,
                                               const SubtitleOptions& options);

protected:
    virtual bool always_include_hours() const = 0;
    virtual std::string decimal_marker() const = 0;

    std::string format_ts(float seconds) const {
        return format_timestamp(seconds, always_include_hours(), decimal_marker());
    }
};

/**
 * GAP 102: WebVTT output with WEBVTT header.
 * Uses "." as decimal marker, hours optional.
 */
class WriteVTT : public SubtitlesWriter {
public:
    using SubtitlesWriter::SubtitlesWriter;

    void write_result(const TranscriptionResult& result, std::ostream& file,
                      const SubtitleOptions& options = SubtitleOptions()) override;

    std::string extension() const override { return "vtt"; }

protected:
    bool always_include_hours() const override { return false; }
    std::string decimal_marker() const override { return "."; }
};

/**
 * GAP 103: SRT output with decimal comma.
 * Uses "," as decimal marker, hours always included.
 */
class WriteSRT : public SubtitlesWriter {
public:
    using SubtitlesWriter::SubtitlesWriter;

    void write_result(const TranscriptionResult& result, std::ostream& file,
                      const SubtitleOptions& options = SubtitleOptions()) override;

    std::string extension() const override { return "srt"; }

protected:
    bool always_include_hours() const override { return true; }
    std::string decimal_marker() const override { return ","; }
};

/**
 * GAP 104: TSV output with integer milliseconds.
 * Tab-separated: start_ms, end_ms, text
 */
class WriteTSV : public ResultWriter {
public:
    using ResultWriter::ResultWriter;

    void write_result(const TranscriptionResult& result, std::ostream& file,
                      const SubtitleOptions& options = SubtitleOptions()) override;

    std::string extension() const override { return "tsv"; }
};

/**
 * GAP 101: JSON output.
 * Full result as JSON object.
 */
class WriteJSON : public ResultWriter {
public:
    using ResultWriter::ResultWriter;

    void write_result(const TranscriptionResult& result, std::ostream& file,
                      const SubtitleOptions& options = SubtitleOptions()) override;

    std::string extension() const override { return "json"; }
};

/**
 * GAP 107: Get writer for output format.
 * Supports "txt", "vtt", "srt", "tsv", "json", "all".
 *
 * @param output_format Format name or "all"
 * @param output_dir Directory to write files to
 * @return Unique pointer to writer, or nullptr if invalid format
 */
std::unique_ptr<ResultWriter> get_writer(const std::string& output_format,
                                          const std::string& output_dir);

/**
 * GAP 107: Write all formats at once.
 * Convenience function that writes txt, vtt, srt, tsv, and json.
 */
void write_all_formats(const TranscriptionResult& result, const std::string& output_dir,
                       const std::string& output_name,
                       const SubtitleOptions& options = SubtitleOptions());

/**
 * Write transcription result to string in specified format.
 * Useful for CLI output without file creation.
 */
std::string write_to_string(const TranscriptionResult& result, const std::string& format,
                            const SubtitleOptions& options = SubtitleOptions());

} // namespace mlx_inference
