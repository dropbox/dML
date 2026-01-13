/**
 * @file python_bindings.cpp
 * @brief Python bindings for SOTA Native C++ Inference
 *
 * Provides Python access to high-performance C++ inference for ECAPA-TDNN and AST.
 * Usage:
 *   import sota_native
 *   ecapa = sota_native.ECAPAInference("weights.safetensors")
 *   ast = sota_native.ASTInference("weights.npz")
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ecapa_inference.h"
#include "ast_inference.h"

namespace py = pybind11;

// Helper to convert numpy array to mlx array
mlx::core::array numpy_to_mlx(py::array_t<float> input) {
    py::buffer_info buf = input.request();

    mlx::core::Shape shape;
    for (auto s : buf.shape) {
        shape.push_back(static_cast<int>(s));
    }

    return mlx::core::array(
        static_cast<float*>(buf.ptr),
        shape,
        mlx::core::float32
    );
}

// Helper to convert mlx array to numpy array
py::array_t<float> mlx_to_numpy(const mlx::core::array& arr) {
    // Evaluate if needed
    mlx::core::eval(arr);

    // Get shape
    std::vector<ssize_t> shape;
    for (int i = 0; i < arr.ndim(); i++) {
        shape.push_back(arr.shape(i));
    }

    // Get data pointer
    auto data = arr.data<float>();

    // Create numpy array
    return py::array_t<float>(shape, data);
}

// Helper to convert mlx array to numpy (int32)
py::array_t<int32_t> mlx_to_numpy_int(const mlx::core::array& arr) {
    mlx::core::eval(arr);

    std::vector<ssize_t> shape;
    for (int i = 0; i < arr.ndim(); i++) {
        shape.push_back(arr.shape(i));
    }

    auto data = arr.data<int32_t>();
    return py::array_t<int32_t>(shape, data);
}

PYBIND11_MODULE(sota_native, m) {
    m.doc() = "SOTA Native C++ Inference - High-performance batch=1 inference for ECAPA-TDNN and AST";

    // ==================== ECAPA-TDNN ====================

    py::class_<ecapa::ECAPAConfig>(m, "ECAPAConfig")
        .def(py::init<>())
        .def_readwrite("n_mels", &ecapa::ECAPAConfig::n_mels)
        .def_readwrite("lin_neurons", &ecapa::ECAPAConfig::lin_neurons)
        .def_readwrite("attention_channels", &ecapa::ECAPAConfig::attention_channels)
        .def_readwrite("res2net_scale", &ecapa::ECAPAConfig::res2net_scale)
        .def_readwrite("se_channels", &ecapa::ECAPAConfig::se_channels)
        .def_readwrite("num_languages", &ecapa::ECAPAConfig::num_languages)
        .def_static("voxlingua107", &ecapa::ECAPAConfig::voxlingua107);

    py::class_<ecapa::ECAPAInference>(m, "ECAPAInference")
        .def(py::init<const std::string&, const ecapa::ECAPAConfig&>(),
             py::arg("weights_path"),
             py::arg("config") = ecapa::ECAPAConfig::voxlingua107())
        .def("load_labels", &ecapa::ECAPAInference::load_labels)
        .def("get_language_code", &ecapa::ECAPAInference::get_language_code)
        .def("compile_model", &ecapa::ECAPAInference::compile_model)
        .def("extract_embedding",
             [](ecapa::ECAPAInference& self, py::array_t<float> mel_input) {
                 auto mlx_input = numpy_to_mlx(mel_input);
                 auto embedding = self.extract_embedding(mlx_input);
                 return mlx_to_numpy(embedding);
             },
             "Extract embeddings from mel features",
             py::arg("mel_input"))
        .def("classify",
             [](ecapa::ECAPAInference& self, py::array_t<float> mel_input) {
                 auto mlx_input = numpy_to_mlx(mel_input);
                 auto [logits, predictions] = self.classify(mlx_input);
                 return py::make_tuple(mlx_to_numpy(logits), mlx_to_numpy(predictions));
             },
             "Classify language from mel features",
             py::arg("mel_input"));

    py::class_<ecapa::ECAPABenchmark>(m, "ECAPABenchmark")
        .def_static("benchmark_latency",
                    &ecapa::ECAPABenchmark::benchmark_latency,
                    py::arg("model"),
                    py::arg("batch_size"),
                    py::arg("seq_len"),
                    py::arg("num_iterations") = 100)
        .def_static("compare_baseline",
                    &ecapa::ECAPABenchmark::compare_baseline,
                    py::arg("model"),
                    py::arg("baseline_ms"),
                    py::arg("batch_size") = 1,
                    py::arg("seq_len") = 300);

    // ==================== AST ====================

    py::class_<ast::ASTConfig>(m, "ASTConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &ast::ASTConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &ast::ASTConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &ast::ASTConfig::num_attention_heads)
        .def_readwrite("intermediate_size", &ast::ASTConfig::intermediate_size)
        .def_readwrite("patch_size", &ast::ASTConfig::patch_size)
        .def_readwrite("num_mel_bins", &ast::ASTConfig::num_mel_bins)
        .def_readwrite("max_length", &ast::ASTConfig::max_length)
        .def_readwrite("time_stride", &ast::ASTConfig::time_stride)
        .def_readwrite("frequency_stride", &ast::ASTConfig::frequency_stride)
        .def_readwrite("num_labels", &ast::ASTConfig::num_labels)
        .def("num_patches", &ast::ASTConfig::num_patches)
        .def_static("audioset", &ast::ASTConfig::audioset);

    py::class_<ast::ASTInference>(m, "ASTInference")
        .def(py::init<const std::string&, const ast::ASTConfig&>(),
             py::arg("weights_path"),
             py::arg("config") = ast::ASTConfig::audioset())
        .def("load_labels", &ast::ASTInference::load_labels)
        .def("get_label", &ast::ASTInference::get_label)
        .def("compile_model", &ast::ASTInference::compile_model)
        .def("extract_features",
             [](ast::ASTInference& self, py::array_t<float> mel_input) {
                 auto mlx_input = numpy_to_mlx(mel_input);
                 auto features = self.extract_features(mlx_input);
                 return mlx_to_numpy(features);
             },
             "Extract pooled features from mel spectrogram",
             py::arg("mel_input"))
        .def("classify",
             [](ast::ASTInference& self, py::array_t<float> mel_input) {
                 auto mlx_input = numpy_to_mlx(mel_input);
                 auto [logits, predictions] = self.classify(mlx_input);
                 return py::make_tuple(mlx_to_numpy(logits), mlx_to_numpy_int(predictions));
             },
             "Classify audio from mel spectrogram",
             py::arg("mel_input"));

    py::class_<ast::ASTBenchmark>(m, "ASTBenchmark")
        .def_static("benchmark_latency",
                    &ast::ASTBenchmark::benchmark_latency,
                    py::arg("model"),
                    py::arg("batch_size"),
                    py::arg("num_iterations") = 100)
        .def_static("compare_baseline",
                    &ast::ASTBenchmark::compare_baseline,
                    py::arg("model"),
                    py::arg("baseline_ms"),
                    py::arg("batch_size") = 1);
}
