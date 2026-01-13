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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "audio_loader.hpp"

namespace nb = nanobind;

NB_MODULE(mlx_audio_native, m) {
    m.doc() = "Native C++ audio loader for WhisperMLX - uses libav (FFmpeg)";

    // Simple function for common use case
    m.def("load_audio",
        [](const std::string& path, int sample_rate) {
            auto samples = mlx_audio::load_audio(path, sample_rate);

            // Allocate new array and copy data
            // nanobind capsule handles memory management
            float* data = new float[samples.size()];
            std::copy(samples.begin(), samples.end(), data);

            nb::capsule owner(data, [](void* p) noexcept {
                delete[] static_cast<float*>(p);
            });

            size_t n = samples.size();
            return nb::ndarray<nb::numpy, float>(
                data,
                {n},
                owner
            );
        },
        nb::arg("path"),
        nb::arg("sample_rate") = 16000,
        R"doc(
Load audio file as float32 numpy array.

Produces identical output to ffmpeg command:
    ffmpeg -nostdin -threads 0 -i <file> -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -

Args:
    path: Path to audio file (wav, mp3, flac, m4a, etc.)
    sample_rate: Target sample rate (default 16000)

Returns:
    numpy.ndarray: Audio samples as float32, normalized to [-1.0, 1.0]

Raises:
    RuntimeError: If file cannot be opened or decoded
)doc"
    );

    // Get audio duration without full decode
    m.def("get_duration",
        &mlx_audio::get_audio_duration,
        nb::arg("path"),
        R"doc(
Get audio duration in seconds without fully decoding.

Args:
    path: Path to audio file

Returns:
    float: Duration in seconds, or -1.0 on error
)doc"
    );

    // Get expected sample count
    m.def("get_sample_count",
        &mlx_audio::get_audio_sample_count,
        nb::arg("path"),
        nb::arg("sample_rate") = 16000,
        R"doc(
Get expected audio sample count at target sample rate.

Args:
    path: Path to audio file
    sample_rate: Target sample rate (default 16000)

Returns:
    int: Estimated sample count, or -1 on error
)doc"
    );

    // AudioLoader class for advanced use
    nb::class_<mlx_audio::AudioLoader>(m, "AudioLoader",
        R"doc(
Audio loader class for repeated loading with same configuration.

Example:
    loader = AudioLoader(sample_rate=16000)
    audio1 = loader.load("file1.wav")
    audio2 = loader.load("file2.mp3")
)doc"
    )
        .def(nb::init<>(), "Create AudioLoader with default config (16kHz mono)")
        .def("__init__",
            [](mlx_audio::AudioLoader* self, int sample_rate) {
                new (self) mlx_audio::AudioLoader(
                    mlx_audio::AudioLoader::Config{sample_rate, 1}
                );
            },
            nb::arg("sample_rate") = 16000,
            "Create AudioLoader with specified sample rate"
        )
        .def("load",
            [](mlx_audio::AudioLoader& self, const std::string& path) {
                auto samples = self.load_float32(path);

                if (!self.ok()) {
                    throw std::runtime_error(self.last_error());
                }

                float* data = new float[samples.size()];
                std::copy(samples.begin(), samples.end(), data);

                nb::capsule owner(data, [](void* p) noexcept {
                    delete[] static_cast<float*>(p);
                });

                size_t n = samples.size();
                return nb::ndarray<nb::numpy, float>(
                    data,
                    {n},
                    owner
                );
            },
            nb::arg("path"),
            "Load audio file as float32 numpy array"
        )
        .def_prop_ro("sample_rate", &mlx_audio::AudioLoader::sample_rate,
            "Target sample rate")
        .def_prop_ro("last_error", &mlx_audio::AudioLoader::last_error,
            "Last error message (empty if no error)")
        .def_prop_ro("ok", &mlx_audio::AudioLoader::ok,
            "True if last operation succeeded");
}
