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

// Kokoro MLX C++ Model Implementation

#include "model.h"
#include "kokoro.h"  // For select_frame_bucket()
#include "simd_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <stdexcept>
#include <optional>
#include <functional>

namespace fs = std::filesystem;

namespace kokoro {

// Helper: Save tensor to numpy format (.npy)
static void save_npy(const std::string& filename, const mx::array& t) {
    mx::eval(t);

    // Get shape
    auto shape = t.shape();
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i < shape.size() - 1 || shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";

    // Create header
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    while ((header.size() + 10) % 64 != 63) header += " ";
    header += "\n";

    std::ofstream file(filename, std::ios::binary);
    if (!file) return;

    // Magic + version
    file.write("\x93NUMPY\x01\x00", 8);
    uint16_t header_len = header.size();
    file.write(reinterpret_cast<char*>(&header_len), 2);
    file.write(header.c_str(), header.size());

    // Data
    size_t num_elements = t.size();
    const float* data = t.data<float>();
    file.write(reinterpret_cast<const char*>(data), num_elements * sizeof(float));
}

// Debug helper: print tensor statistics
static void debug_tensor_stats(const std::string& name, const mx::array& t) {
    if (!std::getenv("DEBUG_GENERATOR")) return;

    mx::eval(t);
    auto flat = mx::reshape(t, {-1});
    mx::eval(flat);
    float min_val = mx::min(flat).item<float>();
    float max_val = mx::max(flat).item<float>();
    float mean_val = mx::mean(flat).item<float>();

    std::cerr << name << ":\n";
    std::cerr << "  shape: [";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << t.shape()[i];
    }
    std::cerr << "]\n";
    std::cerr << "  min: " << min_val << ", max: " << max_val << ", mean: " << mean_val << "\n";

    // Print first 5 values
    int n = std::min(5, (int)flat.size());
    const float* data = flat.data<float>();
    std::cerr << "  first_5: [";
    for (int i = 0; i < n; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << data[i];
    }
    std::cerr << "]\n\n";
}

// Simple JSON parser for config (avoids external dependency)
namespace {

[[maybe_unused]] std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    auto end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract value from JSON for simple key:value pairs
[[maybe_unused]] std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        return json.substr(pos + 1, end - pos - 1);
    }
    return "";
}

int json_get_int(const std::string& json, const std::string& key, int default_val) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    std::string num;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-')) {
        num += json[pos++];
    }
    return num.empty() ? default_val : std::stoi(num);
}

float json_get_float(const std::string& json, const std::string& key, float default_val) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    std::string num;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.')) {
        num += json[pos++];
    }
    return num.empty() ? default_val : std::stof(num);
}

bool json_get_bool(const std::string& json, const std::string& key, bool default_val) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json.substr(pos, 4) == "true") return true;
    if (json.substr(pos, 5) == "false") return false;
    return default_val;
}

std::vector<int> json_get_int_array(const std::string& json, const std::string& key) {
    std::vector<int> result;
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;

    auto end = json.find(']', pos);
    std::string arr = json.substr(pos + 1, end - pos - 1);

    std::string num;
    for (char c : arr) {
        if (isdigit(c) || c == '-') {
            num += c;
        } else if (!num.empty()) {
            result.push_back(std::stoi(num));
            num.clear();
        }
    }
    if (!num.empty()) {
        result.push_back(std::stoi(num));
    }
    return result;
}

}  // anonymous namespace

// =============================================================================
// MLX Helper Functions (used throughout model)
// =============================================================================

// Forward declarations
static mx::array conv1d_dilated(const mx::array& x, const mx::array& weight, const mx::array& bias, int dilation);
static mx::array adain(const mx::array& x, const mx::array& style, const mx::array& fc_weight, const mx::array& fc_bias);
static mx::array conv1d_stride2(const mx::array& x, const mx::array& weight, const mx::array& bias);
static mx::array conv_transpose1d_depthwise_pool(const mx::array& x, const mx::array& weight, const mx::array& bias);
static mx::array adain_resblk1d(const mx::array& x, const mx::array& style, const Weights& weights,
                                const std::string& prefix, bool upsample = false, bool learned_upsample = false);

// Helper: Linear layer
static mx::array linear(const mx::array& x, const mx::array& weight, const mx::array& bias) {
    // Match mlx.nn.Linear: use addmm when bias is present.
    return mx::addmm(bias, x, mx::transpose(weight));
}

// Helper: LayerNorm - match mlx.nn.LayerNorm (mx.fast.layer_norm)
static mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias, float eps = 1e-5f) {
    return mx::fast::layer_norm(
        x,
        std::optional<mx::array>(weight),
        std::optional<mx::array>(bias),
        eps);
}

// Helper: GELU activation - match mlx.nn.gelu (erf-based)
// mlx.nn.gelu: x * (1 + erf(x / sqrt(2))) / 2
static mx::array gelu(const mx::array& x) {
    // Python's mlx.nn.gelu is decorated with mx.compile(shapeless=True).
    // Using mx::compile here matches the compiled-kernel numerics (which differ
    // slightly from the same expression evaluated eagerly).
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            const auto& in = inputs[0];
            constexpr float sqrt2_f32 = 1.4142135381698608f;  // float32(sqrt(2))
            auto y = in * (1.0f + mx::erf(in / sqrt2_f32)) / 2.0f;
            return {y};
        },
        /*shapeless=*/true);
    return compiled({x})[0];
}

// Helper: LeakyReLU activation (compiled for kernel fusion)
static mx::array leaky_relu(const mx::array& x, float negative_slope = 0.2f) {
    // Compile with the common negative_slope=0.2 case
    if (negative_slope == 0.2f) {
        static auto compiled = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                const auto& in = inputs[0];
                return {mx::maximum(in, in * mx::array(0.2f))};
            },
            /*shapeless=*/true);
        return compiled({x})[0];
    }
    // Fallback for other slopes
    return mx::maximum(x, x * mx::array(negative_slope));
}

// Helper: Snake1D activation (crucial for audio synthesis!)
// snake(x) = x + (1/alpha) * sin²(alpha * x)
// alpha is a learnable parameter with shape [1, channels, 1]
// Compiled for kernel fusion and better performance
static mx::array snake1d(const mx::array& x, const mx::array& alpha) {
    // x: [batch, time, channels] in NLC format
    // alpha: [1, channels, 1] -> transpose to [1, 1, channels] for NLC
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            const auto& in = inputs[0];
            const auto& a = inputs[1];
            auto alpha_nlc = mx::transpose(a, {0, 2, 1});  // [1, 1, channels]
            auto ax = in * alpha_nlc;
            auto sin_ax = mx::sin(ax);
            return {in + sin_ax * sin_ax / alpha_nlc};
        },
        /*shapeless=*/true);
    return compiled({x, alpha})[0];
}

// Helper: Conv1d
// MLX conv1d expects:
//   input: [batch, length, in_channels] (NLC format)
//   weight: [out_channels, kernel_size, in_channels]
// Our exported weights are: [out_channels, in_channels, kernel_size]
// So we need to transpose weight from [O, I, K] to [O, K, I]
static mx::array conv1d(const mx::array& x, const mx::array& weight, const mx::array& bias, int padding) {
    // Transpose weight from [out, in, kernel] to [out, kernel, in]
    auto w_transposed = mx::transpose(weight, {0, 2, 1});
    auto out = mx::conv1d(x, w_transposed, 1, padding);  // stride=1
    // Add bias: broadcast [out_channels] to [batch, length, out_channels]
    return out + bias;
}

// Helper: LSTM forward pass (single direction)
// Wx: [4*hidden, input] - input weight
// Wh: [4*hidden, hidden] - hidden weight
// bias: [4*hidden] - combined bias
// Returns: hidden states [batch, length, hidden]
static mx::array lstm_forward(
    const mx::array& x,
    const mx::array& Wx,
    const mx::array& Wh,
    const mx::array& bias
) {
    // Pre-transpose weights to match Python nn.LSTM pattern
    // Python uses: mx.addmm(bias, x, Wx.T) and mx.addmm(ifgo, hidden, Wh.T)
    auto Wx_T = mx::transpose(Wx);
    auto Wh_T = mx::transpose(Wh);

    int batch = x.shape()[0];
    int length = x.shape()[1];
    int hidden_size = Wh.shape()[1];

    // Pre-compute input projections using fused addmm: [batch, length, 4*hidden]
    // Python: x = mx.addmm(self.bias, x, self.Wx.T) which computes bias + x @ Wx.T
    auto x_proj = mx::addmm(bias, x, Wx_T);

    // Initialize hidden state and cell state
    auto h = mx::zeros({batch, hidden_size}, x.dtype());
    auto c = mx::zeros({batch, hidden_size}, x.dtype());

    // Process sequence
    std::vector<mx::array> outputs;
    for (int t = 0; t < length; ++t) {
        // Get input projection for timestep t: [batch, 4*hidden]
        // Use slice instead of take to get [batch, 1, 4*hidden], then reshape
        auto x_t = mx::slice(x_proj, {0, t, 0}, {batch, t + 1, 4 * hidden_size});
        x_t = mx::reshape(x_t, {batch, 4 * hidden_size});  // [batch, 4*hidden]

        // Hidden state projection using fused addmm: [batch, 4*hidden]
        // Python: ifgo = mx.addmm(ifgo, hidden, self.Wh.T) which computes ifgo + hidden @ Wh.T
        auto gates = mx::addmm(x_t, h, Wh_T);

        // Split into 4 gates
        auto i = mx::sigmoid(mx::slice(gates, {0, 0}, {batch, hidden_size}));
        auto f = mx::sigmoid(mx::slice(gates, {0, hidden_size}, {batch, 2 * hidden_size}));
        auto g = mx::tanh(mx::slice(gates, {0, 2 * hidden_size}, {batch, 3 * hidden_size}));
        auto o = mx::sigmoid(mx::slice(gates, {0, 3 * hidden_size}, {batch, 4 * hidden_size}));

        // Update cell and hidden state
        c = f * c + i * g;
        h = o * mx::tanh(c);

        outputs.push_back(mx::expand_dims(h, 1));
    }

    // Stack outputs: [batch, length, hidden]
    return mx::concatenate(outputs, 1);
}

// Helper: BiLSTM forward pass
// Returns: [batch, length, 2*hidden]
static mx::array bilstm_forward(
    const mx::array& x,
    const mx::array& Wx_fwd, const mx::array& Wh_fwd, const mx::array& bias_fwd,
    const mx::array& Wx_bwd, const mx::array& Wh_bwd, const mx::array& bias_bwd
) {
    // Forward pass
    auto out_fwd = lstm_forward(x, Wx_fwd, Wh_fwd, bias_fwd);

    // Backward pass: reverse input, run LSTM, reverse output
    int length = x.shape()[1];
    std::vector<int> reverse_idx;
    for (int i = length - 1; i >= 0; --i) {
        reverse_idx.push_back(i);
    }
    auto x_rev = mx::take(x, mx::array(reverse_idx.data(), {length}), 1);
    auto out_bwd_rev = lstm_forward(x_rev, Wx_bwd, Wh_bwd, bias_bwd);
    auto out_bwd = mx::take(out_bwd_rev, mx::array(reverse_idx.data(), {length}), 1);

    // Concatenate forward and backward outputs
    return mx::concatenate({out_fwd, out_bwd}, -1);
}

// =============================================================================
// KokoroConfig implementation
// =============================================================================

// KokoroConfig implementation
KokoroConfig KokoroConfig::load(const std::string& path) {
    std::string json = read_file(path);
    KokoroConfig config;

    config.dim_in = json_get_int(json, "dim_in", 64);
    config.hidden_dim = json_get_int(json, "hidden_dim", 512);
    config.style_dim = json_get_int(json, "style_dim", 128);
    config.max_conv_dim = json_get_int(json, "max_conv_dim", 512);
    config.n_token = json_get_int(json, "n_token", 178);
    config.n_mels = json_get_int(json, "n_mels", 80);
    config.n_layer = json_get_int(json, "n_layer", 3);
    config.max_dur = json_get_int(json, "max_dur", 50);
    config.dropout = json_get_float(json, "dropout", 0.2f);
    config.text_encoder_kernel_size = json_get_int(json, "text_encoder_kernel_size", 5);
    config.multispeaker = json_get_bool(json, "multispeaker", true);

    config.plbert_hidden_size = json_get_int(json, "plbert_hidden_size", 768);
    config.plbert_num_attention_heads = json_get_int(json, "plbert_num_attention_heads", 12);
    config.plbert_intermediate_size = json_get_int(json, "plbert_intermediate_size", 2048);
    config.plbert_max_position_embeddings = json_get_int(json, "plbert_max_position_embeddings", 512);
    config.plbert_num_hidden_layers = json_get_int(json, "plbert_num_hidden_layers", 12);
    config.plbert_dropout = json_get_float(json, "plbert_dropout", 0.1f);
    config.albert_embedding_dim = json_get_int(json, "albert_embedding_dim", 128);

    config.istft_upsample_rates = json_get_int_array(json, "istft_upsample_rates");
    if (config.istft_upsample_rates.empty()) config.istft_upsample_rates = {10, 6};

    config.istft_upsample_kernel_sizes = json_get_int_array(json, "istft_upsample_kernel_sizes");
    if (config.istft_upsample_kernel_sizes.empty()) config.istft_upsample_kernel_sizes = {20, 12};

    config.istft_gen_istft_n_fft = json_get_int(json, "istft_gen_istft_n_fft", 20);
    config.istft_gen_istft_hop_size = json_get_int(json, "istft_gen_istft_hop_size", 5);
    config.istft_upsample_initial_channel = json_get_int(json, "istft_upsample_initial_channel", 512);

    config.sample_rate = json_get_int(json, "sample_rate", 24000);
    config.hop_size = json_get_int(json, "hop_size", 256);
    config.vocab_size = json_get_int(json, "vocab_size", 178);
    config.bos_token_id = json_get_int(json, "bos_token_id", 0);
    config.eos_token_id = json_get_int(json, "eos_token_id", 0);
    config.weight_norm_folded = json_get_bool(json, "weight_norm_folded", true);

    return config;
}

// Weights implementation
void Weights::load(const std::string& path) {
    auto [loaded_weights, metadata] = mx::load_safetensors(path);
    weights_ = std::move(loaded_weights);
}

mx::array Weights::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool Weights::has(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

std::vector<std::pair<std::string, mx::array>> Weights::get_prefix(const std::string& prefix) const {
    std::vector<std::pair<std::string, mx::array>> result;
    for (const auto& [name, arr] : weights_) {
        if (name.substr(0, prefix.size()) == prefix) {
            result.emplace_back(name, arr);
        }
    }
    return result;
}

// VoicePack implementation
void VoicePack::load(const std::string& path) {
    auto [weights, metadata] = mx::load_safetensors(path);

    // Voice packs have an "embedding" tensor of shape [N, 256]
    // N is typically 1 for single-style voices, or multiple for multi-style
    auto it = weights.find("embedding");
    if (it != weights.end()) {
        auto& voice_tensor = it->second;
        auto shape = voice_tensor.shape();
        int num_lengths = shape[0];

        embeddings_.clear();
        for (int i = 0; i < num_lengths; ++i) {
            embeddings_.push_back(mx::take(voice_tensor, mx::array(i), 0));
        }
    } else {
        throw std::runtime_error("Voice pack missing 'embedding' key");
    }
}

mx::array VoicePack::select(int phoneme_length) const {
    if (embeddings_.empty()) {
        throw std::runtime_error("Voice pack not loaded");
    }
    // Clamp to valid range
    int idx = std::min(phoneme_length - 1, (int)embeddings_.size() - 1);
    idx = std::max(0, idx);
    return embeddings_[idx];
}

// KokoroModel implementation
KokoroModel KokoroModel::load(const std::string& model_path) {
    KokoroModel model;

    // Load config
    fs::path base_path(model_path);
    model.config_ = KokoroConfig::load((base_path / "config.json").string());

    // Load weights
    model.weights_.load((base_path / "weights.safetensors").string());

    // Load available voices
    fs::path voices_dir = base_path / "voices";
    if (fs::exists(voices_dir)) {
        for (const auto& entry : fs::directory_iterator(voices_dir)) {
            if (entry.path().extension() == ".safetensors") {
                std::string voice_name = entry.path().stem().string();
                model.load_voice(voice_name, entry.path().string());
            }
        }
    }

    model.loaded_ = true;
    return model;
}

void KokoroModel::load_voice(const std::string& name, const std::string& path) {
    VoicePack pack;
    pack.load(path);
    voices_[name] = std::move(pack);
}

std::vector<std::string> KokoroModel::available_voices() const {
    std::vector<std::string> result;
    for (const auto& [name, _] : voices_) {
        result.push_back(name);
    }
    return result;
}

bool KokoroModel::has_voice(const std::string& name) const {
    return voices_.find(name) != voices_.end();
}

// Synthesize - partial implementation
mx::array KokoroModel::synthesize(
    const mx::array& tokens,
    const std::string& voice,
    float speed
) {
    // Timing instrumentation (enabled via KOKORO_PROFILE env var)
    static bool profile_enabled = std::getenv("KOKORO_PROFILE") != nullptr;
    auto profile_time = [&](const char* stage) {
        if (profile_enabled) {
            mx::synchronize();
            static auto last_time = std::chrono::high_resolution_clock::now();
            auto now = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - last_time).count();
            std::cerr << "[PROFILE] " << stage << ": " << std::fixed << std::setprecision(1) << ms << " ms\n";
            last_time = now;
        }
    };
    if (profile_enabled) {
        mx::synchronize();
        profile_time("START");
    }

    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    if (!has_voice(voice)) {
        throw std::runtime_error("Voice not found: " + voice);
    }

    // Get voice embedding
    int phoneme_length = tokens.shape()[1] - 2;  // Exclude BOS/EOS
    auto voice_embed = voices_.at(voice).select(phoneme_length);

    // Flatten to [256] if needed (voice_embed might be [1, 256])
    if (voice_embed.ndim() == 2 && voice_embed.shape()[0] == 1) {
        voice_embed = mx::squeeze(voice_embed, 0);  // [256]
    }

    // Split voice embedding into style and speaker parts
    // style: first 128 dims (for decoder)
    // speaker: second 128 dims (for predictor)
    auto style = mx::slice(voice_embed, {0}, {128});      // [128]
    auto speaker = mx::slice(voice_embed, {128}, {256});  // [128]
    profile_time("voice_embed");

    // ====================================================================
    // Step 1: BERT forward pass
    // ====================================================================
    // Match Python no-padding case: attention_mask is None.
    auto empty_mask = mx::zeros({0}, mx::float32);
    auto bert_out = bert_forward(tokens, empty_mask);  // [batch, seq_len, 768]
    profile_time("bert_forward");

    // Save bert_out for debugging (BERT output before linear projection)
    if (std::getenv("DEBUG_F0_TRACE")) {
        save_npy("/tmp/cpp_bert_out.npy", bert_out);
    }

    // bert_encoder: Linear projection from 768 to 512
    auto bert_enc = linear(bert_out,
        weights_.get("bert_encoder.weight"),
        weights_.get("bert_encoder.bias"));  // [batch, seq_len, 512]
    profile_time("bert_encoder");

    // ====================================================================
    // Step 2: Text encoder (for ASR features)
    // ====================================================================
    auto text_enc = text_encoder_forward(tokens);  // [batch, seq_len, 512]
    profile_time("text_encoder");

    // NOTE: Removed mx::eval() calls here - they were forcing premature GPU sync
    // and preventing MLX from optimizing the full computation graph.
    // Only evaluate at the very end for maximum performance.

    if (std::getenv("DEBUG_BERT")) {
        // Only eval for debug output
        mx::eval(bert_enc);
        mx::eval(text_enc);
        std::cerr << "[DEBUG] bert_enc shape: [" << bert_enc.shape()[0] << ", "
                  << bert_enc.shape()[1] << ", " << bert_enc.shape()[2] << "]\n";
        float* data = bert_enc.data<float>();
        std::cerr << "[DEBUG] bert_enc[0,0,:10]: ";
        for (int i = 0; i < 10; i++) {
            std::cerr << data[i] << " ";
        }
        std::cerr << "\n[DEBUG] bert_enc[0,7,:10]: ";
        int offset_7 = 7 * 512;
        for (int i = 0; i < 10; i++) {
            std::cerr << data[offset_7 + i] << " ";
        }
        std::cerr << "\n[DEBUG] bert_enc[0,14,:10]: ";
        int offset_14 = 14 * 512;
        for (int i = 0; i < 10; i++) {
            std::cerr << data[offset_14 + i] << " ";
        }
        std::cerr << "\n";
    }

    // ====================================================================
    // Step 3: Predictor (duration, F0, noise)
    // text_encoder uses style (0:128), F0/N blocks use speaker (128:256)
    // ====================================================================
    auto [asr_features, f0, noise, _unused_style, actual_frames] = predictor_forward(
        bert_enc, text_enc, style, speaker, speed);
    profile_time("predictor");

    // NOTE: Removed mx::eval(asr_features) - let MLX build full graph

    // ====================================================================
    // Step 4: Decoder (audio synthesis)
    // Use style (first 128 dims of voice_embed), NOT speaker!
    // Note: predictor uses bucketed frame sizes for shape-stable compilation
    // We trim the audio to actual_frames * hop_length at the end
    // ====================================================================
    auto style_for_decoder = mx::expand_dims(style, 0);  // [128] -> [1, 128]
    auto audio = decoder_forward(asr_features, f0, noise, style_for_decoder);
    profile_time("decoder");

    // Trim audio to actual length (remove padding from frame bucketing)
    // hop_length = 256, so actual samples = actual_frames * 256
    // Note: audio shape is [1, samples] after decoder_forward
    int actual_samples = actual_frames * 256;
    int audio_samples = audio.shape()[1];  // samples in dim 1
    if (audio_samples > actual_samples) {
        audio = mx::slice(audio, {0, 0}, {1, actual_samples});
    }

    return audio;
}

// BERT embeddings
mx::array KokoroModel::bert_embeddings(const mx::array& input_ids) {
    int batch = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];

    // Word embeddings: [batch, seq_len] -> [batch, seq_len, 128]
    auto word_weight = weights_.get("bert.embeddings.word_embeddings.weight");
    auto word_embeds = mx::take(word_weight, input_ids, 0);

    // Position embeddings: [seq_len] -> [1, seq_len, 128]
    auto pos_weight = weights_.get("bert.embeddings.position_embeddings.weight");
    auto position_ids = mx::astype(mx::arange(seq_len), mx::int32);
    auto pos_embeds = mx::take(pos_weight, position_ids, 0);
    pos_embeds = mx::expand_dims(pos_embeds, 0);  // [1, seq_len, 128]

    // Token type embeddings: zeros -> [batch, seq_len, 128]
    auto type_weight = weights_.get("bert.embeddings.token_type_embeddings.weight");
    auto type_ids = mx::zeros({batch, seq_len}, mx::int32);
    auto type_embeds = mx::take(type_weight, type_ids, 0);

    // Sum embeddings
    auto embeddings = word_embeds + pos_embeds + type_embeds;

    // LayerNorm
    auto ln_weight = weights_.get("bert.embeddings.layer_norm.weight");
    auto ln_bias = weights_.get("bert.embeddings.layer_norm.bias");
    embeddings = layer_norm(embeddings, ln_weight, ln_bias);

    // Note: Projection from 128->768 is done in the ALBERT layer, not embeddings
    // The exported model may not have this separate projection layer
    // For now, return 128-dim embeddings

    return embeddings;  // [batch, seq_len, 128]
}

// ALBERT self-attention layer - uses optimized SDPA kernel
mx::array KokoroModel::albert_attention(const mx::array& hidden, const mx::array& attention_mask) {
    int batch = hidden.shape()[0];
    int seq_len = hidden.shape()[1];
    int hidden_size = config_.plbert_hidden_size;
    int num_heads = config_.plbert_num_attention_heads;
    int head_dim = hidden_size / num_heads;

    // Q, K, V projections
    auto q = linear(hidden,
        weights_.get("bert.encoder.albert_layer.attention.query.weight"),
        weights_.get("bert.encoder.albert_layer.attention.query.bias"));
    auto k = linear(hidden,
        weights_.get("bert.encoder.albert_layer.attention.key.weight"),
        weights_.get("bert.encoder.albert_layer.attention.key.bias"));
    auto v = linear(hidden,
        weights_.get("bert.encoder.albert_layer.attention.value.weight"),
        weights_.get("bert.encoder.albert_layer.attention.value.bias"));

    // Reshape for multi-head attention: [B, T, H*D] -> [B, num_heads, T, head_dim]
    q = mx::transpose(mx::reshape(q, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {batch, seq_len, num_heads, head_dim}), {0, 2, 1, 3});

    // Use optimized SDPA kernel for attention computation
    // Scale is 1/sqrt(head_dim), no causal mask for BERT (bidirectional)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // SDPA with optional attention mask
    auto context = (attention_mask.size() != 0)
        ? mx::fast::scaled_dot_product_attention(q, k, v, scale, "", attention_mask)
        : mx::fast::scaled_dot_product_attention(q, k, v, scale, "", std::nullopt);

    // Reshape back: [B, T, H]
    context = mx::reshape(mx::transpose(context, {0, 2, 1, 3}), {batch, seq_len, hidden_size});

    // Output projection
    auto output = linear(context,
        weights_.get("bert.encoder.albert_layer.attention.dense.weight"),
        weights_.get("bert.encoder.albert_layer.attention.dense.bias"));

    // Residual + LayerNorm
    output = layer_norm(hidden + output,
        weights_.get("bert.encoder.albert_layer.attention.layer_norm.weight"),
        weights_.get("bert.encoder.albert_layer.attention.layer_norm.bias"));

    if (std::getenv("DEBUG_BERT_STEPS")) {
        mx::eval(output);
        float* d = output.data<float>();
        std::cerr << "[DEBUG] After attention [0,0,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
        std::cerr << "\n";
    }

    return output;
}

// ALBERT FFN layer
mx::array KokoroModel::albert_ffn(const mx::array& hidden) {
    static bool saved_ffn0 = false;
    bool should_save = std::getenv("DEBUG_BERT_TRACE") && !saved_ffn0;

    // FFN: hidden -> intermediate -> output
    auto intermediate = linear(hidden,
        weights_.get("bert.encoder.albert_layer.ffn.weight"),
        weights_.get("bert.encoder.albert_layer.ffn.bias"));
    if (should_save) {
        save_npy("/tmp/cpp_ffn_intermediate_0.npy", intermediate);
    }
    intermediate = gelu(intermediate);
    if (should_save) {
        save_npy("/tmp/cpp_ffn_gelu_0.npy", intermediate);
    }

    auto output = linear(intermediate,
        weights_.get("bert.encoder.albert_layer.ffn_output.weight"),
        weights_.get("bert.encoder.albert_layer.ffn_output.bias"));
    if (should_save) {
        save_npy("/tmp/cpp_ffn_linear_out_0.npy", output);
    }

    // Residual + LayerNorm
    output = layer_norm(hidden + output,
        weights_.get("bert.encoder.albert_layer.full_layer_layer_norm.weight"),
        weights_.get("bert.encoder.albert_layer.full_layer_layer_norm.bias"));
    if (should_save) {
        save_npy("/tmp/cpp_ffn_after_ln_0.npy", output);
        saved_ffn0 = true;
    }

    return output;
}

// BERT forward pass
mx::array KokoroModel::bert_forward(const mx::array& tokens, const mx::array& attention_mask) {
    // Get embeddings [batch, seq_len, 128]
    auto hidden = bert_embeddings(tokens);

    if (std::getenv("DEBUG_BERT_TRACE")) {
        save_npy("/tmp/cpp_bert_embeddings.npy", hidden);
    }

    if (std::getenv("DEBUG_BERT_STEPS")) {
        mx::eval(hidden);
        float* d = hidden.data<float>();
        std::cerr << "[DEBUG] After embeddings [0,0,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
        std::cerr << "\n";
    }

    // Project from 128 -> 768 (ALBERT uses shared linear for this)
    // Check if we have the projection weights
    if (weights_.has("bert.encoder.embedding_hidden_mapping_in.weight")) {
        hidden = linear(hidden,
            weights_.get("bert.encoder.embedding_hidden_mapping_in.weight"),
            weights_.get("bert.encoder.embedding_hidden_mapping_in.bias"));
    }

    if (std::getenv("DEBUG_BERT_TRACE")) {
        save_npy("/tmp/cpp_bert_after_projection.npy", hidden);
    }

    if (std::getenv("DEBUG_BERT_STEPS")) {
        mx::eval(hidden);
        float* d = hidden.data<float>();
        std::cerr << "[DEBUG] After projection [0,0,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
        std::cerr << "\n";
    }

    // ALBERT: Same layer repeated num_layers times (weight sharing)
    // Match Python attention mask semantics:
    // - Input mask (if provided as [B, T]) is 1=valid, 0=pad.
    // - Convert to additive mask [B, 1, 1, T]: (1 - mask) * -10000.0
    mx::array extended_mask = attention_mask;
    if (attention_mask.ndim() == 4) {
        // already additive mask
        extended_mask = attention_mask;
    } else if (attention_mask.ndim() == 2 && attention_mask.shape()[1] == tokens.shape()[1]) {
        // [B, T] base mask with 1=valid, 0=pad -> [B, 1, 1, T] additive mask
        extended_mask = mx::expand_dims(mx::expand_dims(attention_mask, 1), 1);
        extended_mask = (mx::array(1.0f) - extended_mask) * mx::array(-10000.0f);
    } else {
        // No padding / no usable mask: represent as empty to match Python's `None`.
        extended_mask = mx::zeros({0}, mx::float32);
    }

    for (int i = 0; i < config_.plbert_num_hidden_layers; ++i) {
        hidden = albert_attention(hidden, extended_mask);
        if (std::getenv("DEBUG_BERT_TRACE") && i == 0) {
            save_npy("/tmp/cpp_bert_after_attn_0.npy", hidden);
        }
        hidden = albert_ffn(hidden);
        if (std::getenv("DEBUG_BERT_TRACE") && i == 0) {
            save_npy("/tmp/cpp_bert_after_ffn_0.npy", hidden);
        }

        // Save first few and last layer outputs to trace error accumulation
        if (std::getenv("DEBUG_BERT_TRACE") && (i < 3 || i == config_.plbert_num_hidden_layers - 1)) {
            save_npy("/tmp/cpp_bert_layer_" + std::to_string(i) + ".npy", hidden);
        }

        if (std::getenv("DEBUG_BERT_STEPS") && i == 0) {
            mx::eval(hidden);
            float* d = hidden.data<float>();
            std::cerr << "[DEBUG] After layer 0 [0,0,:5]: ";
            for (int j = 0; j < 5; j++) std::cerr << d[j] << " ";
            std::cerr << "\n";
        }
    }

    return hidden;  // [batch, seq_len, 768]
}

// Text encoder forward pass
// Takes token IDs and returns encoded features
// Architecture: Embedding -> 3x(Conv1d + LayerNorm + LeakyReLU) -> BiLSTM
mx::array KokoroModel::text_encoder_forward(const mx::array& tokens) {
    // batch and seq_len available for debugging if needed
    (void)tokens.shape()[0];  // batch
    (void)tokens.shape()[1];  // seq_len

    // Step 1: Token embedding [batch, seq_len] -> [batch, seq_len, hidden_dim=512]
    auto embed_weight = weights_.get("text_encoder.embedding.weight");
    auto x = mx::take(embed_weight, tokens, 0);

    // Step 2: Conv1d stack (3 layers)
    // Each layer: Conv1d -> LayerNorm -> LeakyReLU
    int kernel_size = config_.text_encoder_kernel_size;  // 5
    int padding = kernel_size / 2;  // 2

    for (int i = 0; i < 3; ++i) {
        std::string prefix = "text_encoder.convs." + std::to_string(i);

        // Conv1d weights: [out_channels, in_channels, kernel] = [512, 512, 5]
        auto conv_weight = weights_.get(prefix + ".weight");
        auto conv_bias = weights_.get(prefix + ".bias");

        // Apply Conv1d
        x = conv1d(x, conv_weight, conv_bias, padding);

        // LayerNorm (using gamma/beta instead of weight/bias)
        std::string norm_prefix = "text_encoder.norms." + std::to_string(i);
        auto ln_gamma = weights_.get(norm_prefix + ".gamma");
        auto ln_beta = weights_.get(norm_prefix + ".beta");
        x = layer_norm(x, ln_gamma, ln_beta);

        // LeakyReLU
        x = leaky_relu(x, 0.2f);
    }

    // Step 3: BiLSTM
    // Forward LSTM weights
    auto Wx_fwd = weights_.get("text_encoder.lstm.lstm_forward.Wx");
    auto Wh_fwd = weights_.get("text_encoder.lstm.lstm_forward.Wh");
    auto bias_fwd = weights_.get("text_encoder.lstm.lstm_forward.bias");

    // Backward LSTM weights
    auto Wx_bwd = weights_.get("text_encoder.lstm.lstm_backward.Wx");
    auto Wh_bwd = weights_.get("text_encoder.lstm.lstm_backward.Wh");
    auto bias_bwd = weights_.get("text_encoder.lstm.lstm_backward.bias");

    // BiLSTM: [batch, seq_len, 512] -> [batch, seq_len, 512] (256*2)
    x = bilstm_forward(x, Wx_fwd, Wh_fwd, bias_fwd, Wx_bwd, Wh_bwd, bias_bwd);

    return x;  // [batch, seq_len, 512]
}

// Predictor text encoder: style-conditioned BiLSTM stack
// Input: bert_enc [batch, T, 512], style [128]
// Output: duration_feats [batch, T, 640]
mx::array KokoroModel::predictor_text_encoder(
    const mx::array& bert_enc,
    const mx::array& style
) {
    int batch = bert_enc.shape()[0];
    int length = bert_enc.shape()[1];

    // Broadcast style to [batch, T, 128]
    auto style_2d = mx::expand_dims(style, 0);  // [1, 128]
    auto style_3d = mx::broadcast_to(style_2d, {batch, 128});  // [batch, 128]
    auto style_expanded = mx::expand_dims(style_3d, 1);  // [batch, 1, 128]
    auto style_broadcast = mx::broadcast_to(style_expanded, {batch, length, 128});

    // Start with bert_enc
    auto x = bert_enc;  // [batch, T, 512]

    // Process through 6 layers: alternating BiLSTM and AdaLayerNorm
    for (int layer = 0; layer < 6; layer += 2) {
        // Concatenate with style: [batch, T, 640]
        auto x_with_style = mx::concatenate({x, style_broadcast}, -1);

        // BiLSTM layer
        std::string lstm_prefix = "predictor.text_encoder.lstms_" + std::to_string(layer);
        auto Wx_fwd = weights_.get(lstm_prefix + ".lstm_fwd.Wx");
        auto Wh_fwd = weights_.get(lstm_prefix + ".lstm_fwd.Wh");
        auto bias_fwd = weights_.get(lstm_prefix + ".lstm_fwd.bias");
        auto Wx_bwd = weights_.get(lstm_prefix + ".lstm_bwd.Wx");
        auto Wh_bwd = weights_.get(lstm_prefix + ".lstm_bwd.Wh");
        auto bias_bwd = weights_.get(lstm_prefix + ".lstm_bwd.bias");

        x = bilstm_forward(x_with_style, Wx_fwd, Wh_fwd, bias_fwd, Wx_bwd, Wh_bwd, bias_bwd);
        // Output: [batch, T, 512]

        // AdaLayerNorm: x_norm = (1 + gamma) * layer_norm(x) + beta
        // where gamma, beta = fc(style)
        std::string fc_prefix = "predictor.text_encoder.lstms_" + std::to_string(layer + 1);
        auto fc_weight = weights_.get(fc_prefix + ".fc.weight");  // [1024, 128]
        auto fc_bias = weights_.get(fc_prefix + ".fc.bias");      // [1024]

        // Apply fc to style: [batch, 128] -> [batch, 1024]
        auto scale_shift = mx::matmul(style_3d, mx::transpose(fc_weight)) + fc_bias;
        // Split into gamma and beta: each [batch, 512]
        auto gamma = mx::slice(scale_shift, {0, 0}, {batch, 512});
        auto beta = mx::slice(scale_shift, {0, 512}, {batch, 1024});

        // Expand for broadcasting: [batch, 512] -> [batch, 1, 512]
        gamma = mx::expand_dims(gamma, 1);
        beta = mx::expand_dims(beta, 1);

        // Layer norm on x (ddof=0 to match PyTorch LayerNorm)
        auto x_mean = mx::mean(x, -1, true);
        auto diff = x - x_mean;
        int last_dim = x.shape().back();
        auto x_var = mx::sum(diff * diff, -1, true) / mx::array((float)last_dim);  // ddof=0 (population variance)
        auto x_norm = (x - x_mean) / mx::sqrt(x_var + mx::array(1e-5f));

        // AdaLayerNorm: (1 + gamma) * x_norm + beta
        x = (mx::array(1.0f) + gamma) * x_norm + beta;
    }

    // Save bert_enc for comparison (input to predictor_text_encoder)
    if (std::getenv("DEBUG_F0_TRACE")) {
        save_npy("/tmp/cpp_bert_enc.npy", bert_enc);
        save_npy("/tmp/cpp_predictor_text_enc_output.npy", x);  // output before concat
    }

    // Final output: concatenate x with style to get 640-dim
    auto duration_feats = mx::concatenate({x, style_broadcast}, -1);  // [batch, T, 640]

    if (std::getenv("DEBUG_PREDICTOR")) {
        mx::eval(duration_feats);
        float* d = duration_feats.data<float>();
        std::cerr << "[DEBUG] duration_feats[0,0,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
        std::cerr << "\n";
        std::cerr << "[DEBUG] duration_feats[0,7,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[7*640 + i] << " ";
        std::cerr << "\n";
    }

    return duration_feats;
}

// Compute alignment: convert duration logits to frame indices
// Returns: (indices [batch, bucket_size], bucket_size, actual_frames)
// Uses frame bucketing for shape-stable compilation
std::tuple<mx::array, int, int> KokoroModel::compute_alignment(
    const mx::array& duration_logits,
    float speed
) {
    // batch and text_len available for debugging if needed
    (void)duration_logits.shape()[0];  // batch
    int text_len = duration_logits.shape()[1];

    // Duration: sigmoid(logits).sum(-1) / speed
    auto duration = mx::sum(mx::sigmoid(duration_logits), -1) / mx::array(speed);
    auto pred_dur = mx::maximum(mx::round(duration), mx::array(1.0f));

    // Compute total frames
    auto total_frames_arr = mx::sum(pred_dur, -1);  // [batch]
    mx::eval(total_frames_arr);

    // Get max frames (requires sync)
    auto max_frames_data = mx::astype(total_frames_arr, mx::int32);
    mx::eval(max_frames_data);
    int* data = max_frames_data.data<int>();
    int actual_frames = data[0];  // Use first batch element

    // Frame bucketing: round up to nearest bucket for shape-stable compilation
    int bucket_size = select_frame_bucket(actual_frames);

    // Build indices using cumsum (at actual frame count)
    auto cumsum = mx::cumsum(mx::astype(pred_dur, mx::int32), 1);  // [batch, text_len]
    auto frame_pos = mx::arange(actual_frames);  // [actual_frames]
    frame_pos = mx::expand_dims(frame_pos, 0);  // [1, actual_frames]

    // For each frame, find which text position it belongs to
    auto cumsum_exp = mx::expand_dims(cumsum, -1);  // [batch, text_len, 1]
    auto frame_exp = mx::expand_dims(frame_pos, 1);  // [1, 1, actual_frames]

    // comparison: cumsum <= frame_pos
    auto comparison = (cumsum_exp <= frame_exp);  // [batch, text_len, actual_frames]
    auto indices = mx::sum(mx::astype(comparison, mx::int32), 1);  // [batch, actual_frames]
    indices = mx::clip(indices, mx::array(0), mx::array(text_len - 1));

    // Pad indices to bucket_size by repeating last index
    if (bucket_size > actual_frames) {
        int padding = bucket_size - actual_frames;
        // Get the last index value and create padding
        auto last_idx = mx::slice(indices, {0, actual_frames - 1}, {1, actual_frames});  // [1, 1]
        auto pad_indices = mx::broadcast_to(last_idx, {1, padding});  // [1, padding]
        indices = mx::concatenate({indices, pad_indices}, 1);  // [batch, bucket_size]
    }

    return {indices, bucket_size, actual_frames};
}

// Expand features from text rate to audio frame rate
mx::array KokoroModel::expand_features(
    const mx::array& features,
    const mx::array& indices,
    int total_frames
) {
    int batch = features.shape()[0];
    int hidden_dim = features.shape()[2];

    // Expand indices for take_along_axis
    auto indices_exp = mx::expand_dims(indices, -1);  // [batch, total_frames, 1]
    indices_exp = mx::broadcast_to(indices_exp, {batch, total_frames, hidden_dim});

    // Gather features
    return mx::take_along_axis(features, indices_exp, 1);  // [batch, total_frames, hidden_dim]
}

// Predictor forward pass
// Returns: (asr_features, f0, noise, style_128, actual_frames)
// style: voice_embed[0:128] - used for text_encoder
// speaker: voice_embed[128:256] - used for F0/N prediction blocks
// actual_frames: number of actual audio frames (before bucket padding)
std::tuple<mx::array, mx::array, mx::array, mx::array, int> KokoroModel::predictor_forward(
    const mx::array& bert_enc,
    const mx::array& text_enc,
    const mx::array& style,
    const mx::array& speaker,
    float speed
) {
    int batch = bert_enc.shape()[0];
    (void)bert_enc.shape()[1];  // text_len - available for debugging

    // Step 1: predictor.text_encoder (uses speaker, NOT style)
    // PyTorch: duration_feats = model.predictor.text_encoder(d_en, s) where s=voice[128:]
    auto duration_feats = predictor_text_encoder(bert_enc, speaker);  // [batch, T, 640]

    // Step 2: predictor.lstm
    auto lstm_Wx_fwd = weights_.get("predictor.lstm.lstm_fwd.Wx");
    auto lstm_Wh_fwd = weights_.get("predictor.lstm.lstm_fwd.Wh");
    auto lstm_bias_fwd = weights_.get("predictor.lstm.lstm_fwd.bias");
    auto lstm_Wx_bwd = weights_.get("predictor.lstm.lstm_bwd.Wx");
    auto lstm_Wh_bwd = weights_.get("predictor.lstm.lstm_bwd.Wh");
    auto lstm_bias_bwd = weights_.get("predictor.lstm.lstm_bwd.bias");

    auto dur_enc = bilstm_forward(duration_feats,
        lstm_Wx_fwd, lstm_Wh_fwd, lstm_bias_fwd,
        lstm_Wx_bwd, lstm_Wh_bwd, lstm_bias_bwd);  // [batch, T, 512]

    // Step 3: duration_proj
    auto dur_weight = weights_.get("predictor.duration_proj.linear_layer.weight");
    auto dur_bias = weights_.get("predictor.duration_proj.linear_layer.bias");
    auto duration_logits = linear(dur_enc, dur_weight, dur_bias);  // [batch, T, 50]

    // Step 4: compute_alignment
    if (std::getenv("DEBUG_DURATION")) {
        mx::eval(duration_logits);
        std::cerr << "[DEBUG] duration_logits shape: [" << duration_logits.shape()[0]
                  << ", " << duration_logits.shape()[1] << ", " << duration_logits.shape()[2] << "]\n";
        // Print sum of sigmoid for each token (these become per-token durations)
        auto sig_sum = mx::sum(mx::sigmoid(duration_logits), -1);  // [batch, T]
        mx::eval(sig_sum);
        std::cerr << "[DEBUG] per-token sigmoid sums (before speed/round): ";
        float* ss_data = sig_sum.data<float>();
        float total = 0;
        for (int i = 0; i < sig_sum.shape()[1]; i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << ss_data[i];
            total += ss_data[i];
        }
        std::cerr << "\n[DEBUG] Total pre-speed: " << total << ", with speed=" << speed << ": " << (total / speed) << "\n";
    }
    auto [indices, bucket_size, actual_frames] = compute_alignment(duration_logits, speed);
    // Use bucket_size for all shapes (enables kernel caching)
    // actual_frames is used to trim the final audio output
    int total_frames = bucket_size;
    if (std::getenv("DEBUG_DURATION")) {
        std::cerr << "[DEBUG] actual_frames: " << actual_frames
                  << ", bucket_size: " << bucket_size << "\n";
    }

    // Step 5: expand duration_feats to audio frames
    // Save duration_feats BEFORE expand (to trace error source)
    if (std::getenv("DEBUG_F0_TRACE")) {
        save_npy("/tmp/cpp_duration_feats.npy", duration_feats);
        save_npy("/tmp/cpp_indices.npy", indices);
    }

    auto en_expanded = expand_features(duration_feats, indices, total_frames);  // [batch, T_audio, 640]

    // Save BiLSTM input for debug comparison
    if (std::getenv("DEBUG_F0_TRACE")) {
        save_npy("/tmp/cpp_en_expanded_640.npy", en_expanded);
    }

    // Step 6: shared BiLSTM
    auto shared_Wx_fwd = weights_.get("predictor.shared.lstm_fwd.Wx");
    auto shared_Wh_fwd = weights_.get("predictor.shared.lstm_fwd.Wh");
    auto shared_bias_fwd = weights_.get("predictor.shared.lstm_fwd.bias");
    auto shared_Wx_bwd = weights_.get("predictor.shared.lstm_bwd.Wx");
    auto shared_Wh_bwd = weights_.get("predictor.shared.lstm_bwd.Wh");
    auto shared_bias_bwd = weights_.get("predictor.shared.lstm_bwd.bias");

    auto x_shared = bilstm_forward(en_expanded,
        shared_Wx_fwd, shared_Wh_fwd, shared_bias_fwd,
        shared_Wx_bwd, shared_Wh_bwd, shared_bias_bwd);  // [batch, T_audio, 512]

    // Step 7: Expand text_enc to audio frames (for ASR features)
    // PyTorch: asr = text_encoder(input_ids) @ pred_aln_trg (per line 1491-1492 in kokoro.py)
    auto asr_features = expand_features(text_enc, indices, total_frames);  // [batch, T_audio, 512]

    // Step 8: F0 prediction
    // F0 blocks: F0_0 (512->512), F0_1 (512->256 + pool), F0_2 (256->256), F0_proj
    auto x_f0 = x_shared;  // [batch, T_audio, 512]

    if (std::getenv("DEBUG_F0_STEPS")) {
        mx::eval(x_f0);
        float* d = x_f0.data<float>();
        std::cerr << "[DEBUG] x_shared (F0 input) [0,0,:5]: ";
        for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
        std::cerr << "\n";
    }
    if (std::getenv("DEBUG_F0_TRACE")) {
        save_npy("/tmp/cpp_x_shared.npy", x_f0);
    }

    // F0_0: residual block (512 channels)
    // Correct order: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2 -> residual * rsqrt(2)
    {
        auto conv1_w = weights_.get("predictor.F0_0.conv1.weight");  // [512, 512, 3]
        auto conv1_b = weights_.get("predictor.F0_0.conv1.bias");
        auto conv2_w = weights_.get("predictor.F0_0.conv2.weight");
        auto conv2_b = weights_.get("predictor.F0_0.conv2.bias");
        auto norm1_w = weights_.get("predictor.F0_0.norm1.fc.weight");  // [1024, 128]
        auto norm1_b = weights_.get("predictor.F0_0.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.F0_0.norm2.fc.weight");
        auto norm2_b = weights_.get("predictor.F0_0.norm2.fc.bias");

        // Broadcast speaker for AdaLayerNorm: [1, 128] -> [batch, 128]
        // F0/N blocks use speaker (voice_embed[128:256]), NOT style
        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Residual path: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_f0, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);  // [batch, T, 512]
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);

        // Residual with rsqrt(2) scaling per StyleTTS2 upstream
        x_f0 = (h + x_f0) * std::sqrt(0.5f);

        if (std::getenv("DEBUG_F0_STEPS")) {
            mx::eval(x_f0);
            float* d = x_f0.data<float>();
            std::cerr << "[DEBUG] After F0_0 [0,0,:5]: ";
            for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
            std::cerr << "\n";
        }
        if (std::getenv("DEBUG_F0_TRACE")) {
            save_npy("/tmp/cpp_after_F0_0.npy", x_f0);
        }
    }

    // F0_1: UPSAMPLE block (512->256 with 2x upsample)
    // Python: AdainResBlk1d(512, 256, style_dim, upsample=True, learned_upsample=True)
    // learned_upsample uses ConvTranspose1d (pool) in residual path, repeat in shortcut
    {
        auto conv1_w = weights_.get("predictor.F0_1.conv1.weight");  // [256, 512, 3]
        auto conv1_b = weights_.get("predictor.F0_1.conv1.bias");
        auto conv1x1_w = weights_.get("predictor.F0_1.conv1x1.weight");  // [256, 512, 1]
        auto conv1x1_b = weights_.get("predictor.F0_1.conv1x1.bias");
        auto conv2_w = weights_.get("predictor.F0_1.conv2.weight");  // [256, 256, 3]
        auto conv2_b = weights_.get("predictor.F0_1.conv2.bias");
        auto norm1_w = weights_.get("predictor.F0_1.norm1.fc.weight");  // [1024, 128]
        auto norm1_b = weights_.get("predictor.F0_1.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.F0_1.norm2.fc.weight");  // [512, 128]
        auto norm2_b = weights_.get("predictor.F0_1.norm2.fc.bias");
        auto pool_w = weights_.get("predictor.F0_1.pool.weight");  // [512, 1, 3] depthwise
        auto pool_b = weights_.get("predictor.F0_1.pool.bias");    // [512]

        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Shortcut: repeat 2x then project (shortcut always uses repeat even with learned_upsample)
        auto skip = mx::repeat(x_f0, 2, 1);  // [batch, T*2, 512]
        skip = conv1d_dilated(skip, conv1x1_w, conv1x1_b, 1);  // [batch, T*2, 256]

        // Residual path: norm1 -> actv -> pool (ConvTranspose) -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_f0, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv_transpose1d_depthwise_pool(h, pool_w, pool_b);  // learned 2x upsample via ConvTranspose1d
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);  // [batch, T*2, 256]
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);  // [batch, T*2, 256]

        // Residual with rsqrt(2) scaling
        x_f0 = (h + skip) * std::sqrt(0.5f);  // [batch, T*2, 256]

        if (std::getenv("DEBUG_F0_STEPS")) {
            mx::eval(x_f0);
            float* d = x_f0.data<float>();
            std::cerr << "[DEBUG] After F0_1 [0,0,:5]: ";
            for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
            std::cerr << "\n";
        }
        if (std::getenv("DEBUG_F0_TRACE")) {
            save_npy("/tmp/cpp_after_F0_1.npy", x_f0);
        }
    }

    // F0_2: residual block (256 channels)
    // Correct order: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2 -> residual * rsqrt(2)
    {
        auto conv1_w = weights_.get("predictor.F0_2.conv1.weight");  // [256, 256, 3]
        auto conv1_b = weights_.get("predictor.F0_2.conv1.bias");
        auto conv2_w = weights_.get("predictor.F0_2.conv2.weight");
        auto conv2_b = weights_.get("predictor.F0_2.conv2.bias");
        auto norm1_w = weights_.get("predictor.F0_2.norm1.fc.weight");  // [512, 128]
        auto norm1_b = weights_.get("predictor.F0_2.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.F0_2.norm2.fc.weight");
        auto norm2_b = weights_.get("predictor.F0_2.norm2.fc.bias");

        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Residual path: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_f0, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);

        // Residual with rsqrt(2) scaling per StyleTTS2 upstream
        x_f0 = (h + x_f0) * std::sqrt(0.5f);

        if (std::getenv("DEBUG_F0_STEPS")) {
            mx::eval(x_f0);
            float* d = x_f0.data<float>();
            std::cerr << "[DEBUG] After F0_2 [0,0,:5]: ";
            for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
            std::cerr << "\n";
        }
        if (std::getenv("DEBUG_F0_TRACE")) {
            save_npy("/tmp/cpp_after_F0_2.npy", x_f0);
        }
    }

    // F0_proj: [1, 256, 1] -> project to 1 channel
    {
        auto proj_w = weights_.get("predictor.F0_proj.weight");  // [1, 256, 1]
        auto proj_b = weights_.get("predictor.F0_proj.bias");    // [1]
        auto w_trans = mx::transpose(proj_w, {0, 2, 1});  // [1, 1, 256]
        x_f0 = mx::conv1d(x_f0, w_trans, 1, 0) + proj_b;  // [batch, T_audio*2, 1]

        if (std::getenv("DEBUG_F0_STEPS")) {
            mx::eval(x_f0);
            float* d = x_f0.data<float>();
            std::cerr << "[DEBUG] After F0_proj [0,0,:5]: ";
            for (int i = 0; i < 5; i++) std::cerr << d[i] << " ";
            std::cerr << "\n";
        }
    }

    // F0 output: reshape to [batch, T_audio*2]
    auto f0 = mx::reshape(x_f0, {batch, -1});

    // Step 9: Noise prediction (N_0, N_1, N_2, N_proj)
    // Structure mirrors F0 blocks
    auto x_n = x_shared;  // [batch, T_audio, 512]

    // N_0: residual block (512 channels)
    // Correct order: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2 -> residual * rsqrt(2)
    if (weights_.has("predictor.N_0.conv1.weight")) {
        auto conv1_w = weights_.get("predictor.N_0.conv1.weight");
        auto conv1_b = weights_.get("predictor.N_0.conv1.bias");
        auto conv2_w = weights_.get("predictor.N_0.conv2.weight");
        auto conv2_b = weights_.get("predictor.N_0.conv2.bias");
        auto norm1_w = weights_.get("predictor.N_0.norm1.fc.weight");
        auto norm1_b = weights_.get("predictor.N_0.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.N_0.norm2.fc.weight");
        auto norm2_b = weights_.get("predictor.N_0.norm2.fc.bias");

        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Residual path: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_n, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);

        // Residual with rsqrt(2) scaling per StyleTTS2 upstream
        x_n = (h + x_n) * std::sqrt(0.5f);
    }

    // N_1: UPSAMPLE block (512->256 with 2x upsample)
    // Python: AdainResBlk1d(512, 256, style_dim, upsample=True, learned_upsample=True)
    // Same structure as F0_1 - uses ConvTranspose1d (pool) in residual path
    if (weights_.has("predictor.N_1.conv1.weight")) {
        auto conv1_w = weights_.get("predictor.N_1.conv1.weight");
        auto conv1_b = weights_.get("predictor.N_1.conv1.bias");
        auto conv1x1_w = weights_.get("predictor.N_1.conv1x1.weight");
        auto conv1x1_b = weights_.get("predictor.N_1.conv1x1.bias");
        auto conv2_w = weights_.get("predictor.N_1.conv2.weight");
        auto conv2_b = weights_.get("predictor.N_1.conv2.bias");
        auto norm1_w = weights_.get("predictor.N_1.norm1.fc.weight");
        auto norm1_b = weights_.get("predictor.N_1.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.N_1.norm2.fc.weight");
        auto norm2_b = weights_.get("predictor.N_1.norm2.fc.bias");
        auto pool_w = weights_.get("predictor.N_1.pool.weight");  // [512, 1, 3] depthwise
        auto pool_b = weights_.get("predictor.N_1.pool.bias");    // [512]

        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Shortcut: repeat 2x then project (shortcut always uses repeat)
        auto skip = mx::repeat(x_n, 2, 1);
        skip = conv1d_dilated(skip, conv1x1_w, conv1x1_b, 1);

        // Residual path: norm1 -> actv -> pool (ConvTranspose) -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_n, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv_transpose1d_depthwise_pool(h, pool_w, pool_b);  // learned 2x upsample via ConvTranspose1d
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);

        x_n = (h + skip) * std::sqrt(0.5f);
    }

    // N_2: residual block (256 channels)
    // Correct order: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2 -> residual * rsqrt(2)
    if (weights_.has("predictor.N_2.conv1.weight")) {
        auto conv1_w = weights_.get("predictor.N_2.conv1.weight");
        auto conv1_b = weights_.get("predictor.N_2.conv1.bias");
        auto conv2_w = weights_.get("predictor.N_2.conv2.weight");
        auto conv2_b = weights_.get("predictor.N_2.conv2.bias");
        auto norm1_w = weights_.get("predictor.N_2.norm1.fc.weight");
        auto norm1_b = weights_.get("predictor.N_2.norm1.fc.bias");
        auto norm2_w = weights_.get("predictor.N_2.norm2.fc.weight");
        auto norm2_b = weights_.get("predictor.N_2.norm2.fc.bias");

        auto speaker_batch = mx::broadcast_to(mx::expand_dims(speaker, 0), {batch, 128});

        // Residual path: norm1 -> actv -> conv1 -> norm2 -> actv -> conv2
        auto h = adain(x_n, speaker_batch, norm1_w, norm1_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv1_w, conv1_b, 1);
        h = adain(h, speaker_batch, norm2_w, norm2_b);
        h = leaky_relu(h, 0.2f);
        h = conv1d_dilated(h, conv2_w, conv2_b, 1);

        // Residual with rsqrt(2) scaling per StyleTTS2 upstream
        x_n = (h + x_n) * std::sqrt(0.5f);
    }

    // N_proj: project to 1 channel
    mx::array noise = mx::zeros_like(f0);
    if (weights_.has("predictor.N_proj.weight")) {
        auto proj_w = weights_.get("predictor.N_proj.weight");
        auto proj_b = weights_.get("predictor.N_proj.bias");
        auto w_trans = mx::transpose(proj_w, {0, 2, 1});
        x_n = mx::conv1d(x_n, w_trans, 1, 0) + proj_b;
        noise = mx::reshape(x_n, {batch, -1});
    }

    // Style for decoder (first 128 dims)
    auto style_128 = mx::slice(style, {0}, {128});
    style_128 = mx::expand_dims(style_128, 0);  // [1, 128]

    if (std::getenv("DEBUG_F0N")) {
        mx::eval(f0);
        mx::eval(noise);
        float* f0_data = f0.data<float>();
        float* n_data = noise.data<float>();
        std::cerr << "[DEBUG] F0 shape: [" << f0.shape()[0] << ", " << f0.shape()[1] << "]\n";
        std::cerr << "[DEBUG] F0[:10]: ";
        for (int i = 0; i < 10; i++) std::cerr << f0_data[i] << " ";
        std::cerr << "\n";
        std::cerr << "[DEBUG] Noise shape: [" << noise.shape()[0] << ", " << noise.shape()[1] << "]\n";
        std::cerr << "[DEBUG] Noise[:10]: ";
        for (int i = 0; i < 10; i++) std::cerr << n_data[i] << " ";
        std::cerr << "\n";
    }

    return {asr_features, f0, noise, style_128, actual_frames};
}

// Helper: Transpose conv1d for upsampling
// Weight format in export: [in_channels, out_channels, kernel]
// MLX expects: [out_channels, kernel, in_channels]
// IMPORTANT: PyTorch ConvTranspose1d uses padding=(kernel-stride)//2 for proper alignment
static mx::array conv_transpose1d(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    int stride
) {
    // Transpose weight from [in, out, kernel] to [out, kernel, in]
    auto w_trans = mx::transpose(weight, {1, 2, 0});
    int kernel_size = weight.shape()[2];
    int padding = (kernel_size - stride) / 2;
    auto out = mx::conv_transpose1d(x, w_trans, stride, padding);
    return out + bias;
}

// Helper: Depthwise ConvTranspose1d for learned upsampling (pool)
// PyTorch: ConvTranspose1d(in_ch, in_ch, kernel=3, stride=2, groups=in_ch, padding=1, output_padding=1)
// Weight shape: [in_ch, 1, kernel] (depthwise)
// Input: [batch, length, channels]
// Output: [batch, length*2, channels]
static mx::array conv_transpose1d_depthwise_pool(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias
) {
    // Dimensions available for debugging if needed
    (void)x.shape()[0];       // batch
    (void)x.shape()[1];       // input_len
    int channels = x.shape()[2];
    (void)weight.shape()[2];  // kernel_size

    // Transpose weight from [in_ch, 1, kernel] to [out_ch, kernel, in_ch/groups]
    auto w_trans = mx::transpose(weight, {0, 2, 1});  // [channels, kernel, 1]

    // PyTorch: padding=1, output_padding=1 -> output_size = (T-1)*2 - 2*1 + 3 + 1 = 2T
    // MLX C++ signature: conv_transpose1d(input, weight, stride, padding, dilation, output_padding, groups)
    // Use padding=1, output_padding=1 to match Python exactly
    auto out = mx::conv_transpose1d(x, w_trans, 2, 1, 1, 1, channels);  // stride=2, padding=1, dilation=1, output_padding=1, groups=channels

    // Add bias [channels] broadcast to [batch, length, channels]
    return out + bias;
}

// Helper: Conv1d with dilation
// Input: [batch, length, in_channels]
// Weight: [out_channels, in_channels, kernel]
// Returns: [batch, length, out_channels]
static mx::array conv1d_dilated(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    int dilation
) {
    // Transpose weight from [out, in, kernel] to [out, kernel, in]
    auto w_trans = mx::transpose(weight, {0, 2, 1});
    int kernel_size = weight.shape()[2];
    int padding = (kernel_size - 1) * dilation / 2;
    // MLX conv1d signature: conv1d(input, weight, stride, padding, dilation, groups)
    auto out = mx::conv1d(x, w_trans, 1, padding, dilation, 1);
    return out + bias;
}

// Helper: AdaIN (Adaptive Instance Normalization)
// Input x: [batch, length, channels]
// Style: [batch, style_dim=128]
// FC: projects style to [2*channels] for gamma/beta
// Returns: (1 + gamma) * norm(x) + beta
static mx::array adain(
    const mx::array& x,
    const mx::array& style,
    const mx::array& fc_weight,
    const mx::array& fc_bias
) {
    int batch = x.shape()[0];
    (void)x.shape()[1];  // length - available for debugging
    int channels = x.shape()[2];

    // Project style to scale/shift: [batch, 128] -> [batch, 2*channels]
    auto scale_shift = mx::matmul(style, mx::transpose(fc_weight)) + fc_bias;
    auto gamma = mx::slice(scale_shift, {0, 0}, {batch, channels});         // [batch, channels]
    auto beta = mx::slice(scale_shift, {0, channels}, {batch, 2*channels}); // [batch, channels]

    // Expand for broadcasting: [batch, channels] -> [batch, 1, channels]
    gamma = mx::expand_dims(gamma, 1);
    beta = mx::expand_dims(beta, 1);

    // Instance normalization (normalize over length dimension)
    // CRITICAL: PyTorch InstanceNorm1d uses population variance (ddof=0), NOT sample variance!
    // Population variance = sum(x-mean)^2 / N (ddof=0)
    // Sample variance = sum(x-mean)^2 / (N-1) (ddof=1)
    // Using ddof=1 causes ~11% error per layer, accumulating through F0 prediction.
    auto mean = mx::mean(x, 1, true);  // [batch, 1, channels]
    auto diff = x - mean;
    int length_dim = x.shape()[1];
    auto var = mx::sum(diff * diff, 1, true) / mx::array((float)length_dim);  // ddof=0 (population variance)
    auto x_norm = (x - mean) / mx::sqrt(var + mx::array(1e-5f));

    if (std::getenv("DEBUG_ADAIN")) {
        debug_tensor_stats("adain_x_norm", x_norm);
        debug_tensor_stats("adain_gamma", gamma);
        debug_tensor_stats("adain_beta", beta);
    }

    // AdaIN: (1 + gamma) * x_norm + beta
    return (mx::array(1.0f) + gamma) * x_norm + beta;
}

// Helper: Conv1d with stride=2 (for f0_conv and n_conv)
// Input: [batch, length, in_channels]
// Weight: [out_channels, in_channels, kernel]
// Returns: [batch, length/2, out_channels]
static mx::array conv1d_stride2(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias
) {
    // Transpose weight from [out, in, kernel] to [out, kernel, in]
    auto w_trans = mx::transpose(weight, {0, 2, 1});
    int kernel_size = weight.shape()[2];
    int padding = kernel_size / 2;
    // MLX conv1d: conv1d(input, weight, stride, padding, dilation, groups)
    auto out = mx::conv1d(x, w_trans, 2, padding, 1, 1);  // stride=2
    return out + bias;
}

// Helper: AdainResBlk1d - Residual block with AdaIN
// Input: [batch, length, in_channels]
// Style: [batch, style_dim=128]
// Returns: [batch, length (*2 if upsample), out_channels]
static mx::array adain_resblk1d(
    const mx::array& x,
    const mx::array& style,
    const Weights& weights,
    const std::string& prefix,
    bool upsample,
    [[maybe_unused]] bool learned_upsample
) {
    // Get weights
    auto conv1_w = weights.get(prefix + ".conv1.weight");
    auto conv1_b = weights.get(prefix + ".conv1.bias");
    auto conv2_w = weights.get(prefix + ".conv2.weight");
    auto conv2_b = weights.get(prefix + ".conv2.bias");
    auto norm1_fc_w = weights.get(prefix + ".norm1.fc.weight");
    auto norm1_fc_b = weights.get(prefix + ".norm1.fc.bias");
    auto norm2_fc_w = weights.get(prefix + ".norm2.fc.weight");
    auto norm2_fc_b = weights.get(prefix + ".norm2.fc.bias");

    int in_channels = x.shape()[2];
    int out_channels = conv1_w.shape()[0];

    // === Shortcut path ===
    mx::array skip = x;
    if (upsample) {
        // Simple nearest-neighbor upsampling for shortcut
        skip = mx::repeat(x, 2, 1);  // [batch, length*2, channels]
    }

    // Skip projection if channels differ
    if (in_channels != out_channels || weights.has(prefix + ".conv1x1.weight")) {
        auto conv1x1_w = weights.get(prefix + ".conv1x1.weight");
        auto conv1x1_b = weights.get(prefix + ".conv1x1.bias");
        // 1x1 conv: [out, in, 1]
        auto w_trans = mx::transpose(conv1x1_w, {0, 2, 1});
        skip = mx::conv1d(skip, w_trans, 1, 0) + conv1x1_b;
    }

    // === Residual path ===
    // norm1 -> actv -> pool(opt) -> conv1 -> norm2 -> actv -> conv2
    bool debug_encode = std::getenv("DEBUG_ENCODE_BLOCK") && prefix == "decoder.encode";

    if (debug_encode) {
        debug_tensor_stats("encode_input_x", x);
    }

    auto h = adain(x, style, norm1_fc_w, norm1_fc_b);
    if (debug_encode) {
        debug_tensor_stats("encode_after_adain1", h);
    }
    h = leaky_relu(h, 0.2f);
    if (debug_encode) {
        debug_tensor_stats("encode_after_leaky1", h);
    }

    // Pool in residual path - use learned ConvTranspose1d if available
    if (upsample) {
        if (weights.has(prefix + ".pool.weight")) {
            // Learned upsampling via depthwise ConvTranspose1d
            auto pool_w = weights.get(prefix + ".pool.weight");
            auto pool_b = weights.get(prefix + ".pool.bias");
            h = conv_transpose1d_depthwise_pool(h, pool_w, pool_b);
        } else {
            // Simple repeat (matches shortcut behavior)
            h = mx::repeat(h, 2, 1);
        }
    }

    // conv1
    auto w1_trans = mx::transpose(conv1_w, {0, 2, 1});
    int k1 = conv1_w.shape()[2];
    h = mx::conv1d(h, w1_trans, 1, k1/2) + conv1_b;
    if (debug_encode) {
        debug_tensor_stats("encode_after_conv1", h);
    }

    // norm2 -> actv -> conv2
    h = adain(h, style, norm2_fc_w, norm2_fc_b);
    if (debug_encode) {
        debug_tensor_stats("encode_after_adain2", h);
    }
    h = leaky_relu(h, 0.2f);
    if (debug_encode) {
        debug_tensor_stats("encode_after_leaky2", h);
    }

    auto w2_trans = mx::transpose(conv2_w, {0, 2, 1});
    int k2 = conv2_w.shape()[2];
    h = mx::conv1d(h, w2_trans, 1, k2/2) + conv2_b;
    if (debug_encode) {
        debug_tensor_stats("encode_after_conv2", h);
        debug_tensor_stats("encode_skip", skip);
    }

    // Residual with rsqrt(2) scaling
    return (h + skip) * std::sqrt(0.5f);
}

// Helper: Generator resblock with 3 dilations
// Each resblock processes input through 3 dilated conv layers with AdaIN
// Input: [batch, length, channels]
// Style: [batch, 128]
// Returns: [batch, length, channels]
// If custom_prefix is provided, use that instead of "resblocks_{block_idx}"
static mx::array generator_resblock(
    const mx::array& x_in,
    const mx::array& style,
    const Weights& weights,
    int block_idx,
    const std::vector<int>& dilations,  // {1, 3, 5}
    const std::string& custom_prefix = ""
) {
    mx::array x = x_in;  // Will be updated in each iteration
    std::string prefix = custom_prefix.empty()
        ? "decoder.generator.resblocks_" + std::to_string(block_idx)
        : custom_prefix;

    // Process 3 sub-layers with different dilations
    for (int d = 0; d < 3; ++d) {
        int dilation = dilations[d];

        // Get weights for this dilation
        auto conv1_w = weights.get(prefix + ".convs1_" + std::to_string(d) + ".weight");
        auto conv1_b = weights.get(prefix + ".convs1_" + std::to_string(d) + ".bias");
        auto conv2_w = weights.get(prefix + ".convs2_" + std::to_string(d) + ".weight");
        auto conv2_b = weights.get(prefix + ".convs2_" + std::to_string(d) + ".bias");

        auto adain1_fc_w = weights.get(prefix + ".adain1_" + std::to_string(d) + ".fc.weight");
        auto adain1_fc_b = weights.get(prefix + ".adain1_" + std::to_string(d) + ".fc.bias");
        auto adain2_fc_w = weights.get(prefix + ".adain2_" + std::to_string(d) + ".fc.weight");
        auto adain2_fc_b = weights.get(prefix + ".adain2_" + std::to_string(d) + ".fc.bias");

        // Snake1D alpha parameters (NOT for skip connection scaling!)
        auto alpha1 = weights.get(prefix + ".alpha1_" + std::to_string(d));  // [1, channels, 1]
        auto alpha2 = weights.get(prefix + ".alpha2_" + std::to_string(d));  // [1, channels, 1]

        // AdaINResBlock1dStyled computation (per StyleTTS2/BigVGAN):
        // Correct order: AdaIN -> Snake1D -> Conv
        // 1. adain1(x) - instance norm with style conditioning
        // 2. snake1d(xt, alpha1) - periodic activation
        // 3. conv1 (dilated)
        // 4. adain2(xt) - instance norm with style conditioning
        // 5. snake1d(xt, alpha2) - periodic activation
        // 6. conv2 (dilation=1)
        // 7. residual: x = xt + x
        auto xt = adain(x, style, adain1_fc_w, adain1_fc_b);
        bool dbg = (std::getenv("DEBUG_RB0") && block_idx == 0 && d == 0) ||
                   (std::getenv("DEBUG_RB3") && block_idx == 3);
        std::string pf = "rb" + std::to_string(block_idx) + "_d" + std::to_string(d);
        if (dbg) {
            debug_tensor_stats(pf + "_after_adain1", xt);
        }
        xt = snake1d(xt, alpha1);
        if (dbg) {
            debug_tensor_stats(pf + "_after_snake1", xt);
        }
        xt = conv1d_dilated(xt, conv1_w, conv1_b, dilation);
        if (dbg) {
            debug_tensor_stats(pf + "_after_conv1", xt);
        }

        xt = adain(xt, style, adain2_fc_w, adain2_fc_b);
        if (dbg) {
            debug_tensor_stats(pf + "_after_adain2", xt);
        }
        xt = snake1d(xt, alpha2);
        if (dbg) {
            debug_tensor_stats(pf + "_after_snake2", xt);
        }
        xt = conv1d_dilated(xt, conv2_w, conv2_b, 1);  // second conv always dilation=1
        if (dbg) {
            debug_tensor_stats(pf + "_after_conv2", xt);
        }

        // Residual connection
        x = xt + x;
        if (dbg) {
            debug_tensor_stats(pf + "_after_resid", x);
        }
    }

    // WORKAROUND: Divide by 3 to prevent signal explosion
    // Return without internal division - external code divides by num_kernels (3)
    // like Python: xs = sum(resblocks[i](x)) / num_kernels
    return x;
}

// Decoder forward pass with encode/decode blocks + generator
// Full implementation: f0_conv + n_conv + encode + decode_0-3 + generator
mx::array KokoroModel::decoder_forward(
    const mx::array& asr_features,
    const mx::array& f0,
    const mx::array& noise,
    const mx::array& style
) {
    int batch = asr_features.shape()[0];
    int asr_len = asr_features.shape()[1];

    if (std::getenv("DEBUG_DECODE_BLOCKS")) {
        debug_tensor_stats("decoder_input_asr_features", asr_features);
        debug_tensor_stats("decoder_input_f0", f0);
        debug_tensor_stats("decoder_input_noise", noise);
        debug_tensor_stats("decoder_input_style", style);

        // Save to npy files for Python comparison
        save_npy("/tmp/cpp_asr_features.npy", asr_features);
        save_npy("/tmp/cpp_f0.npy", f0);
        save_npy("/tmp/cpp_noise.npy", noise);
        save_npy("/tmp/cpp_style.npy", style);
    }

    // Check required weights exist
    if (!weights_.has("decoder.asr_res.weight") || !weights_.has("decoder.encode.conv1.weight")) {
        int samples = asr_len * 300;
        return mx::zeros({batch, samples}, mx::float32);
    }

    // Store original F0 for generator's m_source
    auto f0_orig = f0;

    // Step 1: Process F0 and noise through stride-2 convolutions
    // Add channel dimension: [batch, length] -> [batch, length, 1]
    auto f0_in = mx::expand_dims(f0, -1);     // [batch, f0_len, 1]
    auto n_in = mx::expand_dims(noise, -1);   // [batch, f0_len, 1]

    auto f0_conv_w = weights_.get("decoder.f0_conv.weight");  // [1, 1, 3]
    auto f0_conv_b = weights_.get("decoder.f0_conv.bias");    // [1]
    auto n_conv_w = weights_.get("decoder.n_conv.weight");    // [1, 1, 3]
    auto n_conv_b = weights_.get("decoder.n_conv.bias");      // [1]

    auto f0_proc = conv1d_stride2(f0_in, f0_conv_w, f0_conv_b);   // [batch, f0_len/2, 1]
    auto n_proc = conv1d_stride2(n_in, n_conv_w, n_conv_b);       // [batch, f0_len/2, 1]

    // Step 2: Process ASR features through asr_res
    auto asr_weight = weights_.get("decoder.asr_res.weight");  // [64, 512, 1]
    auto asr_bias = weights_.get("decoder.asr_res.bias");      // [64]
    auto w_trans = mx::transpose(asr_weight, {0, 2, 1});
    auto asr_res = mx::conv1d(asr_features, w_trans, 1, 0) + asr_bias;  // [batch, asr_len, 64]

    // Step 3: Match lengths between ASR and F0/N processed features
    // F0/N are upsampled 2x relative to ASR by predictor, then downsampled 2x by f0_conv/n_conv
    // So F0/N length should be approximately equal to ASR length
    int f0_len = f0_proc.shape()[1];

    mx::array asr_down = asr_features;
    mx::array asr_res_down = asr_res;
    if (f0_len > asr_len) {
        // Repeat ASR to match F0/N length
        int scale = f0_len / asr_len;
        asr_down = mx::repeat(asr_features, scale, 1);
        asr_down = mx::slice(asr_down, {0, 0, 0}, {batch, f0_len, 512});
        asr_res_down = mx::repeat(asr_res, scale, 1);
        asr_res_down = mx::slice(asr_res_down, {0, 0, 0}, {batch, f0_len, 64});
    } else if (asr_len > f0_len) {
        // Downsample ASR to match F0/N length
        int stride = asr_len / f0_len;
        std::vector<int> indices;
        for (int i = 0; i < f0_len; ++i) {
            indices.push_back(std::min(i * stride, asr_len - 1));
        }
        auto idx_arr = mx::array(indices.data(), {f0_len});
        asr_down = mx::take(asr_features, idx_arr, 1);
        asr_res_down = mx::take(asr_res, idx_arr, 1);
    } else {
        asr_down = asr_features;
        asr_res_down = asr_res;
    }

    // Step 4: Initial concatenation for encode: [batch, length, 512 + 1 + 1] = 514
    auto x = mx::concatenate({asr_down, f0_proc, n_proc}, -1);  // [batch, length, 514]

    // Step 5: Encode block (514 -> 1024)
    x = adain_resblk1d(x, style, weights_, "decoder.encode", false, false);

    if (std::getenv("DEBUG_DECODE_BLOCKS")) {
        debug_tensor_stats("after_encode", x);
        save_npy("/tmp/cpp_after_encode.npy", x);
    }

    // Step 6: Decode blocks with residual concatenation
    // Each decode takes: [x, asr_res_down, f0_proc, n_proc] = [1024 + 64 + 1 + 1] = 1090
    for (int i = 0; i < 4; ++i) {
        std::string prefix = "decoder.decode_" + std::to_string(i);
        bool is_upsample = (i == 3);  // decode_3 has upsample

        // Concatenate residuals: [batch, length, 1090]
        x = mx::concatenate({x, asr_res_down, f0_proc, n_proc}, -1);
        x = adain_resblk1d(x, style, weights_, prefix, is_upsample, is_upsample);

        if (std::getenv("DEBUG_DECODE_BLOCKS")) {
            debug_tensor_stats("after_decode_" + std::to_string(i), x);
            save_npy("/tmp/cpp_after_decode_" + std::to_string(i) + ".npy", x);
        }

        // After upsampling block, adjust residual lengths
        if (is_upsample) {
            int new_len = x.shape()[1];
            asr_res_down = mx::repeat(asr_res_down, 2, 1);
            asr_res_down = mx::slice(asr_res_down, {0, 0, 0}, {batch, new_len, 64});
            f0_proc = mx::repeat(f0_proc, 2, 1);
            f0_proc = mx::slice(f0_proc, {0, 0, 0}, {batch, new_len, 1});
            n_proc = mx::repeat(n_proc, 2, 1);
            n_proc = mx::slice(n_proc, {0, 0, 0}, {batch, new_len, 1});
        }
    }
    // x is now [batch, frames*2, 512] after decode_3 upsample
    debug_tensor_stats("decoder_output_to_generator", x);

    // Step 7: Generator upsampling
    int frames = x.shape()[1];
    if (!weights_.has("decoder.generator.ups_0.weight")) {
        int samples = frames * 150;  // 10 * 6 * 5 / 2 (already upsampled 2x)
        return mx::zeros({batch, samples}, mx::float32);
    }

    // Generate harmonic source from F0
    // F0 is [batch, F0_frames] where F0_frames = T_audio / 2
    // Need to upsample and generate harmonics
    int total_upp = 10 * 6 * 5;  // = 300 (upsample_rates * istft_hop_size)
    int f0_orig_frames = f0_orig.shape()[1];

    // Upsample F0 to sample rate: f0_frames * total_upp samples
    int audio_samples = f0_orig_frames * total_upp;

    // Generate harmonic source using m_source
    // For each sample: generate 9 harmonics, weight them with l_linear
    int stft_n_fft = 20;
    int stft_hop = 5;
    int stft_pad = stft_n_fft / 2;
    int padded_len = audio_samples + 2 * stft_pad;
    int stft_frames = (padded_len - stft_n_fft) / stft_hop + 1;
    mx::array source = mx::zeros({1, stft_frames, 22}, mx::float32);

    if (weights_.has("decoder.generator.m_source.l_linear.weight")) {
        if (frames != f0_orig_frames) {
            throw std::runtime_error(
                "Generator mismatch: x frames (" + std::to_string(frames) +
                ") != f0 frames (" + std::to_string(f0_orig_frames) + ")."
            );
        }

        auto l_weight = weights_.get("decoder.generator.m_source.l_linear.weight");  // [1, 9]
        auto l_bias = weights_.get("decoder.generator.m_source.l_linear.bias");      // [1]

        // High-performance path with anti-aliasing interpolation matching Python SourceModule:
        // Reference: tools/pytorch_to_mlx/converters/models/kokoro.py::SourceModule.__call__()
        // 1. Upsample F0 to audio rate
        // 2. Compute rad_values for all harmonics
        // 3. Downsample for anti-aliasing (linear interp)
        // 4. Cumsum at low rate
        // 5. Scale by upp
        // 6. Upsample back (linear interp)
        // 7. Apply sin with amplitude
        const float pi = 3.14159265358979323846f;
        const float sample_rate = 24000.0f;
        const float sine_amp = 0.1f;
        const int num_harmonics = 9;
        int length = f0_orig_frames;  // Original F0 frames

        // DO NOT clamp negative F0 - Python SourceModule doesn't clamp!
        // Negative F0 values contribute to phase accumulation in rad_values.
        // Only the UV mask (F0 > 10 Hz) controls voiced/unvoiced output.

        // Upsample F0 to audio rate: [B, length] -> [B, samples]
        auto f0_rep = mx::repeat(mx::expand_dims(f0_orig, -1), total_upp, 2);
        auto f0_upsampled = mx::reshape(f0_rep, {batch, audio_samples});

        // Voiced mask (F0 > 10 Hz)
        auto uv = mx::astype(mx::greater(f0_upsampled, mx::array(10.0f)), mx::float32);  // [B, samples]

        // Harmonic multipliers: [1, 2, ..., 9] -> [1, 1, 9]
        auto h_factors = mx::reshape(mx::arange(1, num_harmonics + 1, mx::float32), {1, 1, num_harmonics});

        // Compute rad_values for all harmonics: [B, samples, 9]
        // rad_values = (f0 * h / sample_rate) % 1
        auto f0_expanded = mx::expand_dims(f0_upsampled, -1);  // [B, samples, 1]
        auto rad_values = mx::remainder(f0_expanded * h_factors / mx::array(sample_rate), mx::array(1.0f));

        // Downsample for anti-aliasing: [B, samples, 9] -> [B, length, 9]
        // t_down = (arange(length) + 0.5) * samples / length - 0.5, clamped to [0, samples-1]
        auto idx_raw = (mx::arange(length, mx::float32) + mx::array(0.5f)) * mx::array((float)audio_samples / (float)length) - mx::array(0.5f);
        auto t_down = mx::clip(idx_raw, mx::array(0.0f), mx::array((float)(audio_samples - 1)));
        auto t_floor_down = mx::astype(mx::floor(t_down), mx::int32);
        auto t_ceil_down = mx::minimum(t_floor_down + mx::array(1), mx::array(audio_samples - 1));
        auto t_frac_down = t_down - mx::astype(t_floor_down, mx::float32);

        // Linear interpolation: rad_values_down = rad_floor * (1 - frac) + rad_ceil * frac
        auto rad_floor = mx::take(rad_values, t_floor_down, 1);  // [B, length, 9]
        auto rad_ceil = mx::take(rad_values, t_ceil_down, 1);
        auto t_frac_3d = mx::reshape(t_frac_down, {1, length, 1});  // [1, length, 1]
        auto rad_values_down = rad_floor * (mx::array(1.0f) - t_frac_3d) + rad_ceil * t_frac_3d;

        // Cumsum at low rate: [B, length, 9]
        // Float32 cumsum - precision is acceptable after phase fix
        auto phase_cumsum = mx::cumsum(rad_values_down, 1);
        auto two_pi = mx::array(2.0f * pi);
        auto phase_low = phase_cumsum * two_pi;

        // Scale by upp: [B, length, 9]
        auto phase_scaled = phase_low * mx::array((float)total_upp);

        // Upsample back: [B, length, 9] -> [B, samples, 9]
        auto idx_raw_up = (mx::arange(audio_samples, mx::float32) + mx::array(0.5f)) * mx::array((float)length / (float)audio_samples) - mx::array(0.5f);
        auto t_up = mx::clip(idx_raw_up, mx::array(0.0f), mx::array((float)(length - 1)));
        auto t_floor_up = mx::astype(mx::floor(t_up), mx::int32);
        auto t_ceil_up = mx::minimum(t_floor_up + mx::array(1), mx::array(length - 1));
        auto t_frac_up = t_up - mx::astype(t_floor_up, mx::float32);

        auto phase_floor = mx::take(phase_scaled, t_floor_up, 1);  // [B, samples, 9]
        auto phase_ceil = mx::take(phase_scaled, t_ceil_up, 1);
        auto t_frac_up_3d = mx::reshape(t_frac_up, {1, audio_samples, 1});  // [1, samples, 1]
        auto phase = phase_floor * (mx::array(1.0f) - t_frac_up_3d) + phase_ceil * t_frac_up_3d;

        // Apply sin with amplitude: [B, samples, 9]
        auto harmonics_stack = mx::sin(phase) * mx::array(sine_amp);

        // UV mask expanded: [B, samples, 1]
        auto uv_expanded = mx::expand_dims(uv, -1);

        // Apply UV mask (voiced regions only have harmonics, unvoiced = 0)
        // In deterministic mode, noise is 0, so sine_waves = harmonics * uv
        auto sine_waves = harmonics_stack * uv_expanded;

        // Combine harmonics using l_linear + tanh: [B, samples, 9] -> [B, samples, 1] -> [B, samples]
        // l_linear: [1, 9], l_bias: [1]
        auto w = mx::reshape(l_weight, {1, 1, num_harmonics});  // [1, 1, 9]
        auto combined = mx::sum(sine_waves * w, -1) + mx::reshape(l_bias, {1, 1});  // [B, samples]
        auto harmonic = mx::tanh(combined);  // [B, audio_samples]

        debug_tensor_stats("harmonic_signal", harmonic);
        if (std::getenv("SAVE_DEBUG_TENSORS")) {
            save_npy("/tmp/cpp_har_source.npy", harmonic);
        }

        // Reflect pad harmonic signal (without repeating edge).
        // Left indices: [pad, pad-1, ..., 1]
        // Right indices: [L-2, L-3, ..., L-pad-1]
        auto i0 = mx::astype(mx::arange(stft_pad), mx::int32);  // [pad]
        auto left_idx = mx::astype(mx::array(stft_pad), mx::int32) - i0;  // [pad..1]
        auto right_idx = mx::astype(mx::array(audio_samples - 2), mx::int32) - i0;  // [L-2..L-pad-1]

        auto left_pad = mx::take(harmonic, left_idx, 1);
        auto right_pad = mx::take(harmonic, right_idx, 1);
        auto harmonic_padded = mx::concatenate({left_pad, harmonic, right_pad}, 1);  // [B, padded_len]

        // Frame the signal: indices = arange(n_fft) + arange(frames)*hop
        auto frame_base = mx::expand_dims(mx::astype(mx::arange(stft_frames), mx::int32) * mx::array(stft_hop), 1);
        auto frame_offs = mx::expand_dims(mx::astype(mx::arange(stft_n_fft), mx::int32), 0);
        auto idx = frame_base + frame_offs;  // [stft_frames, stft_n_fft]
        auto idx_flat = mx::reshape(idx, {-1});

        auto frames_flat = mx::take(harmonic_padded, idx_flat, 1);
        auto framed = mx::reshape(frames_flat, {batch, stft_frames, stft_n_fft});

        // Hann window.
        auto n = mx::astype(mx::arange(stft_n_fft), mx::float32);
        auto window = mx::array(0.5f) * (mx::array(1.0f) - mx::cos(mx::array(2.0f * pi / stft_n_fft) * n));
        framed = framed * mx::reshape(window, {1, 1, stft_n_fft});

        // RFFT to get 11 complex bins, then magnitude+phase.
        auto spec_c = mx::fft::rfft(framed, stft_n_fft, -1);  // [B, stft_frames, 11] complex
        auto mag = mx::abs(spec_c);
        // Add +0.0f to convert -0.0 to +0.0, matching Python arctan2 behavior
        auto spec_real = mx::real(spec_c) + mx::array(0.0f);
        auto spec_imag = mx::imag(spec_c) + mx::array(0.0f);
        auto ph = mx::arctan2(spec_imag, spec_real);

        // Zero phase where magnitude is small (phase is arbitrary/noisy there)
        // This ensures consistent phase values between C++ and Python when FFT bins
        // have small magnitudes where tiny residual differences cause large phase changes.
        // Threshold 0.01 eliminates nearly all phase wrapping issues.
        constexpr float mag_eps = 0.01f;
        ph = mx::where(mag < mx::array(mag_eps), mx::zeros_like(ph), ph);

        source = mx::concatenate({mag, ph}, -1);  // [B, stft_frames, 22]
    }
    // else: source is already initialized to zeros above

    debug_tensor_stats("source_stft", source);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_source_stft.npy", source);
    }

    x = leaky_relu(x, 0.1f);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_generator_input.npy", x);
    }
    auto ups0_w = weights_.get("decoder.generator.ups_0.weight");  // [512, 256, 20]
    auto ups0_b = weights_.get("decoder.generator.ups_0.bias");    // [256]
    x = conv_transpose1d(x, ups0_w, ups0_b, 10);  // [batch, frames*10, 256]
    debug_tensor_stats("after_ups_0", x);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_after_ups0_only.npy", x);
    }

    // Dilations for resblocks and noise_res blocks
    std::vector<int> dilations = {1, 3, 5};

    // Process source through noise_convs_0 and noise_res_0, add to x
    // Per StyleTTS2: noise_convs_0 has stride=prod(upsample_rates[1:])=6, kernel=12, padding=(6+1)//2=3
    if (weights_.has("decoder.generator.noise_convs_0.weight")) {
        auto nc0_w = weights_.get("decoder.generator.noise_convs_0.weight");  // [256, 22, 12]
        auto nc0_b = weights_.get("decoder.generator.noise_convs_0.bias");

        // noise_convs_0: kernel=12, stride=6, padding=3
        auto w_nc = mx::transpose(nc0_w, {0, 2, 1});  // [256, 12, 22]
        auto x_source = mx::conv1d(source, w_nc, 6, 3) + nc0_b;  // stride=6, padding=3
        debug_tensor_stats("x_source_after_noise_conv_0", x_source);
        if (std::getenv("SAVE_DEBUG_TENSORS")) {
            save_npy("/tmp/cpp_x_source_after_nc0.npy", x_source);
        }

        // noise_res_0: AdaINResBlock1dStyled (kernel=7, 256 channels)
        // Process x_source through noise_res_0 with style conditioning
        if (weights_.has("decoder.generator.noise_res_0.convs1_0.weight")) {
            x_source = generator_resblock(x_source, style, weights_, -1, dilations,
                "decoder.generator.noise_res_0");  // Use -1 for custom prefix
            debug_tensor_stats("x_source_after_noise_res_0", x_source);
            if (std::getenv("SAVE_DEBUG_TENSORS")) {
                save_npy("/tmp/cpp_x_source_after_nr0.npy", x_source);
            }
        }

        if (x_source.shape()[1] != x.shape()[1]) {
            throw std::runtime_error(
                "Generator length mismatch at stage 0: x=" + std::to_string(x.shape()[1]) +
                " x_source=" + std::to_string(x_source.shape()[1]) +
                " (expected equal; check source STFT + noise_conv_0 stride/padding)"
            );
        }

        // Add source to main path
        x = x + x_source;
    }

    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_after_ups_0.npy", x);
    }

    // Apply resblocks 0, 1, 2 (256 channels, kernel sizes 3, 7, 11)
    // Python: xs = sum(resblocks[i](x)) / num_kernels
    debug_tensor_stats("before_resblocks_012", x);
    auto rb0 = generator_resblock(x, style, weights_, 0, dilations);
    debug_tensor_stats("rb0_output", rb0);
    auto rb1 = generator_resblock(x, style, weights_, 1, dilations);
    debug_tensor_stats("rb1_output", rb1);
    auto rb2 = generator_resblock(x, style, weights_, 2, dilations);
    debug_tensor_stats("rb2_output", rb2);
    x = (rb0 + rb1 + rb2) / mx::array(3.0f);
    debug_tensor_stats("after_resblocks_012", x);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_after_resblocks_0.npy", x);
    }

    // ups_1: stride=6, kernel=12, [256, 128, 12] -> in_channels=256, out_channels=128
    x = leaky_relu(x, 0.1f);

    // Process source through noise_convs_1 BEFORE upsampling (per StyleTTS2)
    // noise_convs_1: kernel=1, stride=1, padding=0
    mx::array x_source_1 = mx::zeros({batch, 1, 128}, mx::float32);
    if (weights_.has("decoder.generator.noise_convs_1.weight")) {
        auto nc1_w = weights_.get("decoder.generator.noise_convs_1.weight");  // [128, 22, 1]
        auto nc1_b = weights_.get("decoder.generator.noise_convs_1.bias");

        // noise_convs_1: kernel=1, stride=1, padding=0
        auto w_nc = mx::transpose(nc1_w, {0, 2, 1});  // [128, 1, 22]
        x_source_1 = mx::conv1d(source, w_nc, 1, 0) + nc1_b;  // stride=1, padding=0
        debug_tensor_stats("x_source_after_noise_conv_1", x_source_1);

        // noise_res_1: AdaINResBlock1dStyled (kernel=11, 128 channels)
        // Process x_source_1 through noise_res_1 with style conditioning
        if (weights_.has("decoder.generator.noise_res_1.convs1_0.weight")) {
            x_source_1 = generator_resblock(x_source_1, style, weights_, -1, dilations,
                "decoder.generator.noise_res_1");
            debug_tensor_stats("x_source_after_noise_res_1", x_source_1);
        }
    }

    auto ups1_w = weights_.get("decoder.generator.ups_1.weight");  // [256, 128, 12]
    auto ups1_b = weights_.get("decoder.generator.ups_1.bias");    // [128]
    x = conv_transpose1d(x, ups1_w, ups1_b, 6);  // [batch, frames*60, 128]
    debug_tensor_stats("after_ups_1", x);

    // Reflect pad (last stage): prepend x[:, 1:2, :]
    (void)x.shape()[1];  // out_len - available for debugging
    auto pad_val = mx::slice(x, {0, 1, 0}, {batch, 2, 128});
    x = mx::concatenate({pad_val, x}, 1);  // [batch, frames*60+1, 128]

    if (x_source_1.shape()[1] != x.shape()[1]) {
        throw std::runtime_error(
            "Generator length mismatch at stage 1: x=" + std::to_string(x.shape()[1]) +
            " x_source=" + std::to_string(x_source_1.shape()[1]) +
            " (expected equal; check source STFT framing and stage-1 reflect pad)"
        );
    }
    x = x + x_source_1;
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_after_ups_1.npy", x);
    }

    // Apply resblocks 3, 4, 5 (128 channels, kernel sizes 3, 7, 11)
    // Python: xs = sum(resblocks[i](x)) / num_kernels
    debug_tensor_stats("before_resblocks_345", x);
    if (std::getenv("SAVE_RB_TENSORS")) {
        save_npy("/tmp/cpp_before_rb345.npy", x);
        save_npy("/tmp/cpp_generator_style.npy", style);
    }
    auto rb3 = generator_resblock(x, style, weights_, 3, dilations);
    debug_tensor_stats("rb3_output", rb3);
    auto rb4 = generator_resblock(x, style, weights_, 4, dilations);
    debug_tensor_stats("rb4_output", rb4);
    auto rb5 = generator_resblock(x, style, weights_, 5, dilations);
    debug_tensor_stats("rb5_output", rb5);
    x = (rb3 + rb4 + rb5) / mx::array(3.0f);
    debug_tensor_stats("after_resblocks_345", x);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_after_resblocks_1.npy", x);
    }

    // Step 3: conv_post to get spectrogram
    // conv_post: [22, 128, 7] kernel=7, padding=3
    x = leaky_relu(x, 0.01f);  // PyTorch nn.LeakyReLU() default is 0.01
    debug_tensor_stats("before_conv_post", x);
    if (std::getenv("SAVE_DEBUG_TENSORS")) {
        save_npy("/tmp/cpp_before_conv_post.npy", x);
    }
    auto post_w = weights_.get("decoder.generator.conv_post.weight");  // [22, 128, 7]
    auto post_b = weights_.get("decoder.generator.conv_post.bias");    // [22]
    auto w_post = mx::transpose(post_w, {0, 2, 1});
    auto spec = mx::conv1d(x, w_post, 1, 3) + post_b;  // [batch, T, 22]
    debug_tensor_stats("conv_post_output", spec);
    if (std::getenv("SAVE_ISTFT_INPUT")) {
        save_npy("/tmp/cpp_conv_post_output.npy", spec);
    }

    // Step 4: ISTFT conversion using MLX FFT
    // Split into magnitude (11) and phase (11)
    int n_bins = 11;  // n_fft=20 -> 11 bins
    auto log_mag = mx::slice(spec, {0, 0, 0}, {batch, (int)spec.shape()[1], n_bins});
    auto phase_logits = mx::slice(spec, {0, 0, n_bins}, {batch, (int)spec.shape()[1], 22});

    // Magnitude: exp of log-magnitude
    debug_tensor_stats("log_mag", log_mag);
    debug_tensor_stats("phase_logits", phase_logits);
    auto mag = mx::exp(log_mag);
    // Phase: sin of phase logits to bound to [-1, 1]
    auto phase = mx::sin(phase_logits);
    debug_tensor_stats("istft_mag", mag);
    debug_tensor_stats("istft_phase", phase);

    // ISTFT parameters
    int n_fft = config_.istft_gen_istft_n_fft;  // 20
    int hop_size = config_.istft_gen_istft_hop_size;  // 5
    int spec_frames = spec.shape()[1];

    // Construct complex spectrum from magnitude and phase
    // Python: spectrum = mag * exp(1j * phase) where phase = sin(phase_logits) in [-1, 1] radians
    // exp(1j * phase) = cos(phase) + i*sin(phase)
    auto cos_phase = mx::cos(phase);
    auto sin_phase = mx::sin(phase);

    // Real and imaginary parts: mag * cos(phase), mag * sin(phase)
    auto real_part = mag * cos_phase;
    auto imag_part = mag * sin_phase;

    // Construct complex spectrum: real + i*imag
    auto spectrum = mx::astype(real_part, mx::complex64) +
                    mx::multiply(mx::array(std::complex<float>(0.0f, 1.0f)),
                                mx::astype(imag_part, mx::complex64));

    // Use MLX irfft for each frame
    auto time_frames = mx::fft::irfft(spectrum, n_fft, -1);  // [batch, frames, n_fft]
    debug_tensor_stats("time_frames", time_frames);

    // Create Hann window using SIMD utility
    std::vector<float> window_data(n_fft);
    simd::generate_hann_window_neon(window_data.data(), n_fft);
    auto window = mx::array(window_data.data(), {1, 1, n_fft}, mx::float32);

    // Precompute window squared for overlap-add normalization
    std::vector<float> window_squared(n_fft);
    simd::elementwise_multiply_neon(window_squared.data(), window_data.data(), window_data.data(), n_fft);

    // Apply window
    time_frames = time_frames * window;

    // Overlap-add synthesis with SIMD optimization
    int output_length = (spec_frames - 1) * hop_size;
    mx::eval(time_frames);

    const float* frame_data = time_frames.data<float>();
    std::vector<float> audio(output_length + n_fft, 0.0f);
    std::vector<float> window_sum(output_length + n_fft, 0.0f);

    // Use SIMD-optimized overlap-add
    for (int t = 0; t < spec_frames; ++t) {
        int start = t * hop_size;
        simd::overlap_add_neon(
            audio.data(),
            window_sum.data(),
            frame_data + t * n_fft,
            window_squared.data(),
            n_fft,
            start,
            static_cast<int>(audio.size())
        );
    }

    // Normalize by window sum using SIMD
    simd::normalize_neon(audio.data(), window_sum.data(), static_cast<int>(audio.size()), 1e-8f);

    // Trim padding (center=True equivalent)
    int pad = n_fft / 2;
    int final_len = std::min(output_length, (int)audio.size() - pad);
    std::vector<float> trimmed(audio.begin() + pad, audio.begin() + pad + final_len);

    return mx::array(trimmed.data(), {1, (int)trimmed.size()}, mx::float32);
}

}  // namespace kokoro
