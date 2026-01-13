# C++ Voice System - Implementation Guide

This document provides detailed implementation guidance for the key components of the C++ voice system.

---

## 1. Main Pipeline Architecture

### Core Pipeline Class

```cpp
// stream-tts-cpp/src/voice_pipeline.hpp
#pragma once

#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#include <queue>
#include <memory>
#include <atomic>

class VoicePipeline {
public:
    explicit VoicePipeline(const VoiceSystemConfig& config)
        : config_(config)
        , io_context_()
        , classifier_(config)
        , summarizer_(config.summarization.model_path)
        , translator_(config.translation.model_path)
        , tts_engine_(config.tts)
        , audio_player_(config.audio)
        , voice_queue_(config.queue)
        , running_(false)
    {
        init_components();
    }

    // Main processing loop
    void run() {
        running_ = true;

        // Start worker threads
        std::thread classifier_thread([this] { classification_worker(); });
        std::thread translation_thread([this] { translation_worker(); });
        std::thread tts_thread([this] { tts_worker(); });
        std::thread playback_thread([this] { playback_worker(); });

        // Main thread: read stdin and parse JSON
        stdin_reader();

        // Cleanup
        running_ = false;
        classifier_thread.join();
        translation_thread.join();
        tts_thread.join();
        playback_thread.join();
    }

    void stop() {
        running_ = false;
        voice_queue_.shutdown();
    }

private:
    // Stage 1: Read JSON from stdin
    void stdin_reader() {
        std::string line;
        while (running_ && std::getline(std::cin, line)) {
            if (line.empty()) continue;

            try {
                auto json = nlohmann::json::parse(line);
                raw_message_queue_.push(json);
                raw_message_cv_.notify_one();
            } catch (const nlohmann::json::exception& e) {
                // Not JSON, skip
                if (config_.logging.level <= spdlog::level::debug) {
                    spdlog::debug("Skipping non-JSON line: {}", e.what());
                }
            }
        }
    }

    // Stage 2: Classify messages
    void classification_worker() {
        while (running_) {
            auto msg = wait_and_pop(raw_message_queue_, raw_message_cv_);
            if (!msg) continue;

            auto classification = classifier_.classify(*msg);

            if (classification.priority == MessagePriority::SKIP) {
                continue;
            }

            // If needs summarization, send to summarizer
            if (classification.should_summarize) {
                summarization_queue_.push({*msg, classification});
                summarization_cv_.notify_one();
            } else {
                // Direct to translation
                translation_queue_.push({
                    classification.priority,
                    classification.content,
                    std::chrono::steady_clock::now()
                });
                translation_cv_.notify_one();
            }
        }
    }

    // Stage 3: LLM Summarization (parallel to classification)
    void summarization_worker() {
        while (running_) {
            auto item = wait_and_pop(summarization_queue_, summarization_cv_);
            if (!item) continue;

            auto& [msg, classification] = *item;

            // Generate summary
            auto summary = summarizer_.summarize(msg);

            // Send to translation
            translation_queue_.push({
                classification.priority,
                summary,
                std::chrono::steady_clock::now()
            });
            translation_cv_.notify_one();
        }
    }

    // Stage 4: Translation
    void translation_worker() {
        while (running_) {
            auto item = wait_and_pop(translation_queue_, translation_cv_);
            if (!item) continue;

            std::string translated;
            if (config_.translation.enabled) {
                translated = translator_.translate(item->content);
            } else {
                translated = item->content;
            }

            // Send to TTS
            tts_queue_.push({
                item->priority,
                translated,
                item->created_at
            });
            tts_cv_.notify_one();
        }
    }

    // Stage 5: TTS Generation
    void tts_worker() {
        while (running_) {
            auto item = wait_and_pop(tts_queue_, tts_cv_);
            if (!item) continue;

            // Generate audio
            auto audio_data = tts_engine_.synthesize(item->content);

            // Send to playback
            playback_queue_.push({
                item->priority,
                std::move(audio_data),
                item->created_at
            });
            playback_cv_.notify_one();
        }
    }

    // Stage 6: Audio Playback
    void playback_worker() {
        while (running_) {
            auto item = wait_and_pop(playback_queue_, playback_cv_);
            if (!item) continue;

            // Check latency
            auto now = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - item->created_at
            ).count();

            if (latency > config_.queue.max_latency_ms) {
                spdlog::warn("Dropping audio segment due to latency: {}ms > {}ms",
                           latency, config_.queue.max_latency_ms);
                if (config_.logging.log_dropped_messages) {
                    // Log dropped message
                }
                continue;
            }

            // Play audio
            audio_player_.play(item->audio_data, item->priority);

            if (config_.logging.log_latencies) {
                spdlog::info("Played audio with {}ms latency", latency);
            }
        }
    }

    // Helper: wait and pop from queue
    template<typename T>
    std::optional<T> wait_and_pop(std::queue<T>& queue,
                                   std::condition_variable& cv) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv.wait(lock, [&] { return !queue.empty() || !running_; });

        if (!running_ && queue.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue.front());
        queue.pop();
        return item;
    }

    void init_components() {
        spdlog::set_level(config_.logging.level);
        spdlog::info("Initializing voice pipeline...");

        // Pre-load models
        summarizer_.warmup();
        translator_.warmup();
        tts_engine_.warmup();

        spdlog::info("Voice pipeline ready");
    }

    VoiceSystemConfig config_;
    boost::asio::io_context io_context_;

    MessageClassifier classifier_;
    ToolCallSummarizer summarizer_;
    TranslationEngine translator_;
    TTSEngine tts_engine_;
    AudioPlayer audio_player_;
    PriorityVoiceQueue voice_queue_;

    // Pipeline queues
    std::queue<nlohmann::json> raw_message_queue_;
    std::queue<std::pair<nlohmann::json, Classification>> summarization_queue_;
    std::queue<TextSegment> translation_queue_;
    std::queue<TextSegment> tts_queue_;
    std::queue<AudioSegment> playback_queue_;

    // Synchronization
    std::mutex queue_mutex_;
    std::condition_variable raw_message_cv_;
    std::condition_variable summarization_cv_;
    std::condition_variable translation_cv_;
    std::condition_variable tts_cv_;
    std::condition_variable playback_cv_;

    std::atomic<bool> running_;
};
```

---

## 2. LLM Summarizer Implementation

### Tool Call Parser

```cpp
// src/tool_call_parser.hpp
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

struct ParsedToolCall {
    std::string tool_name;
    std::unordered_map<std::string, std::string> params;
    std::string context;  // Surrounding conversation context

    std::string to_prompt_format() const {
        std::ostringstream oss;
        oss << "Tool: " << tool_name << "\n";
        oss << "Parameters:\n";
        for (const auto& [key, value] : params) {
            oss << "  - " << key << ": " << value << "\n";
        }
        return oss.str();
    }
};

class ToolCallParser {
public:
    ParsedToolCall parse(const nlohmann::json& msg) {
        ParsedToolCall result;

        // Extract tool name
        if (msg.contains("content_block") &&
            msg["content_block"].contains("name")) {
            result.tool_name = msg["content_block"]["name"];
        }

        // Extract parameters
        if (msg.contains("content_block") &&
            msg["content_block"].contains("input")) {
            auto input = msg["content_block"]["input"];
            result.params = flatten_params(input);
        }

        return result;
    }

private:
    std::unordered_map<std::string, std::string> flatten_params(
        const nlohmann::json& input
    ) {
        std::unordered_map<std::string, std::string> result;

        for (auto& [key, value] : input.items()) {
            if (value.is_string()) {
                result[key] = value.get<std::string>();
            } else if (value.is_number()) {
                result[key] = std::to_string(value.get<double>());
            } else if (value.is_boolean()) {
                result[key] = value.get<bool>() ? "true" : "false";
            } else {
                result[key] = value.dump();  // JSON string for complex types
            }
        }

        return result;
    }
};
```

### LLM Integration with llama.cpp

```cpp
// src/tool_summarizer.cpp
#include "tool_summarizer.hpp"
#include "llama.h"
#include <spdlog/spdlog.h>

class ToolCallSummarizer::Impl {
public:
    Impl(const std::string& model_path) {
        // Initialize llama.cpp
        llama_backend_init();

        // Load model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;  // Use all GPU layers
        model_params.use_mlock = true;    // Lock model in RAM

        model_ = llama_load_model_from_file(model_path.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("Failed to load model: " + model_path);
        }

        // Create context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;      // Small context for summaries
        ctx_params.n_batch = 128;
        ctx_params.n_threads = 4;
        ctx_params.n_threads_batch = 4;

        ctx_ = llama_new_context_with_model(model_, ctx_params);
        if (!ctx_) {
            llama_free_model(model_);
            throw std::runtime_error("Failed to create context");
        }

        // Create sampler
        auto sparams = llama_sampler_chain_default_params();
        smpl_ = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl_,
            llama_sampler_init_temp(0.3f));  // Low temperature for consistency
        llama_sampler_chain_add(smpl_,
            llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl_,
            llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        spdlog::info("LLM summarizer loaded: {}", model_path);
    }

    ~Impl() {
        if (smpl_) llama_sampler_free(smpl_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_free_model(model_);
        llama_backend_free();
    }

    std::string generate(const std::string& prompt) {
        // Tokenize prompt
        std::vector<llama_token> tokens;
        tokens.resize(prompt.size() + 256);  // Reserve space

        int n_tokens = llama_tokenize(
            model_,
            prompt.c_str(),
            prompt.size(),
            tokens.data(),
            tokens.size(),
            true,   // add_special
            false   // parse_special
        );

        if (n_tokens < 0) {
            spdlog::error("Failed to tokenize prompt");
            return "";
        }

        tokens.resize(n_tokens);

        // Decode prompt tokens
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

        if (llama_decode(ctx_, batch) != 0) {
            spdlog::error("Failed to decode prompt");
            return "";
        }

        // Generate tokens
        std::string result;
        const int max_tokens = 30;  // Short summaries only

        for (int i = 0; i < max_tokens; ++i) {
            auto token = llama_sampler_sample(smpl_, ctx_, -1);

            // Check for EOS
            if (llama_token_is_eog(model_, token)) {
                break;
            }

            // Decode token to text
            char buf[256];
            int len = llama_token_to_piece(model_, token, buf, sizeof(buf), 0, false);

            if (len > 0) {
                result.append(buf, len);
            }

            // Prepare next batch with single token
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx_, batch) != 0) {
                spdlog::error("Failed to decode token");
                break;
            }
        }

        return result;
    }

    void warmup() {
        // Run a dummy generation to warm up the model
        generate("Test");
        llama_kv_cache_clear(ctx_);
    }

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* smpl_ = nullptr;
};

// Public interface
ToolCallSummarizer::ToolCallSummarizer(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path))
{
}

ToolCallSummarizer::~ToolCallSummarizer() = default;

std::string ToolCallSummarizer::summarize(const nlohmann::json& tool_call) {
    auto parsed = parser_.parse(tool_call);

    // Check cache
    std::string cache_key = create_cache_key(parsed);
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    // Build prompt
    std::string prompt = build_prompt(parsed);

    // Generate summary
    auto summary = impl_->generate(prompt);

    // Clean up summary
    summary = trim(summary);

    // Cache result
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[cache_key] = summary;

        // Simple LRU: evict oldest if too large
        if (cache_.size() > max_cache_size_) {
            cache_.erase(cache_.begin());
        }
    }

    return summary;
}

std::string ToolCallSummarizer::build_prompt(const ParsedToolCall& tool) {
    // Optimized prompt for Qwen2.5-Instruct format
    std::ostringstream prompt;

    prompt << "<|im_start|>system\n"
           << "You describe coding actions in first person present continuous tense. "
           << "Be concise (10-15 words). Examples:\n"
           << "- Read main.cpp → \"I'm reading main.cpp to understand the entry point\"\n"
           << "- Grep \"class Parser\" → \"I'm searching for the Parser class definition\"\n"
           << "- Bash git status → \"I'm checking the git status\"\n"
           << "<|im_end|>\n"
           << "<|im_start|>user\n"
           << "Describe this action:\n"
           << tool.to_prompt_format()
           << "<|im_end|>\n"
           << "<|im_start|>assistant\n";

    return prompt.str();
}

void ToolCallSummarizer::warmup() {
    impl_->warmup();
}
```

---

## 3. Smart Tool Call Batching

### Batch Detector

```cpp
// src/tool_batch_detector.hpp
#pragma once

#include <vector>
#include <chrono>
#include <string>

enum class ToolPattern {
    SINGLE,              // Single tool call
    MULTIPLE_READS,      // Multiple Read calls
    GREP_THEN_READ,      // Grep followed by Read
    MULTI_EDIT,          // Multiple Edit calls
    BASH_SEQUENCE,       // Multiple Bash commands
    MIXED               // No clear pattern
};

struct ToolBatch {
    ToolPattern pattern;
    std::vector<ParsedToolCall> tools;
    std::chrono::steady_clock::time_point first_call;
    std::chrono::steady_clock::time_point last_call;

    std::string summarize() const {
        using namespace std::chrono;

        auto duration = duration_cast<milliseconds>(last_call - first_call);

        switch (pattern) {
            case ToolPattern::SINGLE:
                return "";  // Will be individually summarized

            case ToolPattern::MULTIPLE_READS: {
                size_t count = tools.size();
                if (count == 2) {
                    return "I'm reading two files to understand the code structure";
                } else if (count <= 5) {
                    return fmt::format("I'm reading {} files to analyze the codebase", count);
                } else {
                    return fmt::format("I'm scanning {} files to gather context", count);
                }
            }

            case ToolPattern::GREP_THEN_READ: {
                // Extract grep pattern
                std::string pattern_str = "code";
                if (!tools.empty() && tools[0].params.contains("pattern")) {
                    pattern_str = tools[0].params["pattern"];
                    // Simplify regex patterns
                    pattern_str = simplify_pattern(pattern_str);
                }
                return fmt::format("I'm searching for {} and reading the results",
                                 pattern_str);
            }

            case ToolPattern::MULTI_EDIT: {
                return fmt::format("I'm making {} code changes", tools.size());
            }

            case ToolPattern::BASH_SEQUENCE: {
                return "I'm running several commands to complete the task";
            }

            case ToolPattern::MIXED:
            default:
                return fmt::format("I'm performing {} actions", tools.size());
        }
    }

private:
    static std::string simplify_pattern(const std::string& regex) {
        // Convert common regex patterns to plain English
        // e.g., "class.*Parser" -> "Parser class"
        //       "import.*json" -> "json imports"

        std::string result = regex;

        // Remove regex metacharacters for simple patterns
        static const std::vector<std::pair<std::string, std::string>> replacements = {
            {".*", " "},
            {"\\s+", " "},
            {"\\w+", "word"},
            {"[", ""},
            {"]", ""},
            {"+", ""},
            {"*", ""},
        };

        for (const auto& [from, to] : replacements) {
            size_t pos = 0;
            while ((pos = result.find(from, pos)) != std::string::npos) {
                result.replace(pos, from.length(), to);
                pos += to.length();
            }
        }

        return trim(result);
    }
};

class ToolBatchDetector {
public:
    explicit ToolBatchDetector(int window_ms = 1000)
        : batch_window_(std::chrono::milliseconds(window_ms))
    {
    }

    void add_tool(const ParsedToolCall& tool) {
        auto now = std::chrono::steady_clock::now();

        // Start new batch if needed
        if (current_batch_.tools.empty()) {
            current_batch_.first_call = now;
            current_batch_.tools.push_back(tool);
            return;
        }

        // Check if within window
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - current_batch_.last_call
        );

        if (elapsed > batch_window_) {
            // Flush current batch
            flush_batch();

            // Start new batch
            current_batch_.first_call = now;
            current_batch_.tools.push_back(tool);
        } else {
            // Add to current batch
            current_batch_.tools.push_back(tool);
        }

        current_batch_.last_call = now;
    }

    std::optional<ToolBatch> get_batch() {
        if (pending_batches_.empty()) {
            return std::nullopt;
        }

        auto batch = pending_batches_.front();
        pending_batches_.pop();
        return batch;
    }

    void flush() {
        if (!current_batch_.tools.empty()) {
            flush_batch();
        }
    }

private:
    void flush_batch() {
        // Detect pattern
        current_batch_.pattern = detect_pattern(current_batch_.tools);

        // Add to pending
        pending_batches_.push(current_batch_);

        // Reset
        current_batch_ = ToolBatch{};
    }

    ToolPattern detect_pattern(const std::vector<ParsedToolCall>& tools) {
        if (tools.size() == 1) {
            return ToolPattern::SINGLE;
        }

        // Check for all same tool
        bool all_read = std::all_of(tools.begin(), tools.end(),
            [](const auto& t) { return t.tool_name == "Read"; });

        if (all_read) {
            return ToolPattern::MULTIPLE_READS;
        }

        bool all_edit = std::all_of(tools.begin(), tools.end(),
            [](const auto& t) { return t.tool_name == "Edit"; });

        if (all_edit) {
            return ToolPattern::MULTI_EDIT;
        }

        bool all_bash = std::all_of(tools.begin(), tools.end(),
            [](const auto& t) { return t.tool_name == "Bash"; });

        if (all_bash) {
            return ToolPattern::BASH_SEQUENCE;
        }

        // Check for Grep -> Read pattern
        if (tools.size() >= 2 &&
            tools[0].tool_name == "Grep" &&
            tools[1].tool_name == "Read") {
            return ToolPattern::GREP_THEN_READ;
        }

        return ToolPattern::MIXED;
    }

    ToolBatch current_batch_;
    std::queue<ToolBatch> pending_batches_;
    std::chrono::milliseconds batch_window_;
};
```

---

## 4. Configuration System

### YAML Config Loading

```cpp
// src/config.cpp
#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <cxxopts.hpp>
#include <fstream>

VoiceSystemConfig VoiceSystemConfig::load(const std::string& config_path) {
    VoiceSystemConfig config;

    try {
        YAML::Node yaml = YAML::LoadFile(config_path);

        // Filtering
        if (yaml["filtering"]) {
            auto f = yaml["filtering"];
            config.filtering.min_priority =
                parse_priority(f["min_priority"].as<std::string>("MEDIUM"));
            config.filtering.min_text_length =
                f["min_text_length"].as<size_t>(10);
            config.filtering.speak_tool_calls =
                f["speak_tool_calls"].as<bool>(true);
            config.filtering.speak_tool_results =
                f["speak_tool_results"].as<bool>(false);
            config.filtering.speak_thinking =
                f["speak_thinking"].as<bool>(false);
        }

        // Summarization
        if (yaml["summarization"]) {
            auto s = yaml["summarization"];
            config.summarization.enabled = s["enabled"].as<bool>(true);
            config.summarization.model_path =
                s["model_path"].as<std::string>();
            config.summarization.max_tokens = s["max_tokens"].as<int>(30);
            config.summarization.temperature = s["temperature"].as<float>(0.3f);
            config.summarization.cache_size = s["cache_size"].as<size_t>(100);
            config.summarization.batch_similar_tools =
                s["batch_similar_tools"].as<bool>(true);
            config.summarization.batch_window_ms =
                s["batch_window_ms"].as<int>(1000);
        }

        // Queue
        if (yaml["queue"]) {
            auto q = yaml["queue"];
            config.queue.strategy =
                parse_queue_strategy(q["strategy"].as<std::string>("INTERRUPT_HIGH"));
            config.queue.max_queue_size = q["max_queue_size"].as<size_t>(50);
            config.queue.max_latency_ms = q["max_latency_ms"].as<size_t>(10000);
            config.queue.allow_interrupts = q["allow_interrupts"].as<bool>(true);
        }

        // Translation
        if (yaml["translation"]) {
            auto t = yaml["translation"];
            config.translation.enabled = t["enabled"].as<bool>(true);
            config.translation.source_lang = t["source_lang"].as<std::string>("en");
            config.translation.target_lang = t["target_lang"].as<std::string>("ja");
            config.translation.model_path = t["model_path"].as<std::string>();
        }

        // TTS
        if (yaml["tts"]) {
            auto tts = yaml["tts"];
            config.tts.engine = tts["engine"].as<std::string>("piper");
            config.tts.voice = tts["voice"].as<std::string>();
            config.tts.speed = tts["speed"].as<float>(1.0f);
        }

        spdlog::info("Loaded config from: {}", config_path);

    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        throw;
    }

    return config;
}

VoiceSystemConfig VoiceSystemConfig::from_cli(int argc, char** argv) {
    cxxopts::Options options("stream-tts-cpp",
                            "Intelligent voice feedback for Claude Code");

    options.add_options()
        ("c,config", "Config file path",
         cxxopts::value<std::string>()->default_value(""))
        ("l,lang", "Target language (en, ja, es, etc.)",
         cxxopts::value<std::string>()->default_value("ja"))
        ("no-translate", "Disable translation")
        ("no-summarize", "Disable tool call summarization")
        ("voice", "TTS voice name",
         cxxopts::value<std::string>())
        ("speed", "Speech speed multiplier",
         cxxopts::value<float>()->default_value("1.0"))
        ("model", "LLM model for summarization",
         cxxopts::value<std::string>())
        ("log-level", "Logging level (debug, info, warn, error)",
         cxxopts::value<std::string>()->default_value("info"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // Load from config file if provided
    VoiceSystemConfig config;
    if (result.count("config") && !result["config"].as<std::string>().empty()) {
        config = load(result["config"].as<std::string>());
    }

    // Override with CLI args
    if (result.count("lang")) {
        config.translation.target_lang = result["lang"].as<std::string>();
        config.translation.enabled = (config.translation.target_lang != "en");
    }

    if (result.count("no-translate")) {
        config.translation.enabled = false;
    }

    if (result.count("no-summarize")) {
        config.summarization.enabled = false;
    }

    if (result.count("voice")) {
        config.tts.voice = result["voice"].as<std::string>();
    }

    if (result.count("speed")) {
        config.tts.speed = result["speed"].as<float>();
    }

    if (result.count("model")) {
        config.summarization.model_path = result["model"].as<std::string>();
    }

    if (result.count("log-level")) {
        config.logging.level = parse_log_level(result["log-level"].as<std::string>());
    }

    return config;
}
```

---

## 5. Performance Monitoring

### Latency Tracking

```cpp
// src/performance_monitor.hpp
#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <spdlog/spdlog.h>

class PerformanceMonitor {
public:
    struct Metrics {
        size_t count = 0;
        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;

        double avg_ms() const {
            return count > 0 ? total_ms / count : 0.0;
        }

        void record(double latency_ms) {
            count++;
            total_ms += latency_ms;
            min_ms = std::min(min_ms, latency_ms);
            max_ms = std::max(max_ms, latency_ms);
        }
    };

    class ScopedTimer {
    public:
        ScopedTimer(PerformanceMonitor& monitor, const std::string& stage)
            : monitor_(monitor)
            , stage_(stage)
            , start_(std::chrono::steady_clock::now())
        {
        }

        ~ScopedTimer() {
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start_
            ).count() / 1000.0;  // Convert to ms

            monitor_.record(stage_, duration);
        }

    private:
        PerformanceMonitor& monitor_;
        std::string stage_;
        std::chrono::steady_clock::time_point start_;
    };

    void record(const std::string& stage, double latency_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_[stage].record(latency_ms);
    }

    void print_stats() {
        std::lock_guard<std::mutex> lock(mutex_);

        spdlog::info("=== Performance Statistics ===");
        for (const auto& [stage, metrics] : metrics_) {
            spdlog::info("{:20} count={:5} avg={:6.1f}ms min={:6.1f}ms max={:6.1f}ms",
                       stage,
                       metrics.count,
                       metrics.avg_ms(),
                       metrics.min_ms,
                       metrics.max_ms);
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
    }

private:
    std::unordered_map<std::string, Metrics> metrics_;
    std::mutex mutex_;
};

// Usage:
// {
//     auto timer = perf_monitor.start("llm_summarization");
//     auto summary = summarizer.summarize(tool_call);
//     // Timer automatically records when it goes out of scope
// }
```

---

## 6. Example Main Function

```cpp
// src/main.cpp
#include "voice_pipeline.hpp"
#include "config.hpp"
#include <csignal>
#include <memory>

std::unique_ptr<VoicePipeline> g_pipeline;

void signal_handler(int signal) {
    if (g_pipeline) {
        spdlog::info("Shutting down gracefully...");
        g_pipeline->stop();
    }
    exit(0);
}

int main(int argc, char** argv) {
    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Load configuration
        auto config = VoiceSystemConfig::from_cli(argc, argv);

        // Initialize logging
        spdlog::set_level(config.logging.level);
        spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

        spdlog::info("Starting Stream TTS C++ v1.0.0");
        spdlog::info("Target language: {}", config.translation.target_lang);
        spdlog::info("TTS engine: {}", config.tts.engine);
        spdlog::info("LLM summarization: {}", config.summarization.enabled ? "ON" : "OFF");

        // Create and run pipeline
        g_pipeline = std::make_unique<VoicePipeline>(config);
        g_pipeline->run();

        spdlog::info("Shutdown complete");
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}
```

---

## Build Instructions

```bash
# Clone repository
git clone https://github.com/dropbox/dML/voice
cd voice/stream-tts-cpp

# Install dependencies (macOS)
brew install cmake boost yaml-cpp spdlog

# Clone submodules
git submodule update --init --recursive

# Build llama.cpp with Metal support
cd external/llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release
cd ../..

# Build piper-tts
cd external/piper
cmake -B build
cmake --build build --config Release
cd ../..

# Build main project
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run
./build/stream-tts-cpp --config config.yaml
```

---

This implementation provides a production-ready, high-performance C++ voice system with intelligent message filtering and LLM-powered summarization. The modular design allows easy customization and extension for different use cases.
