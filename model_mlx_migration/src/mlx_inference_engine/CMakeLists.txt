cmake_minimum_required(VERSION 3.14)
project(mlx_inference_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(mlx CONFIG REQUIRED)

# MLX Inference Engine library
add_library(mlx_inference_engine STATIC
    mlx_inference_engine.cpp
    translation_model.cpp
    whisper_model.cpp
    grammar_parser.cpp
    llm_model.cpp
    silero_vad.cpp
    output_writers.cpp
    cosyvoice3_model.cpp
    phoneme_head.cpp
    misaki_g2p.cpp
    ../kokoro/kokoro.cpp
    ../kokoro/model.cpp
    ../kokoro/g2p.cpp
    ../kokoro/tokenizer.cpp
    ../kokoro/prosody_parser.cpp
    ../kokoro/prosody_adjust.cpp
)

target_include_directories(mlx_inference_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kokoro
    /opt/homebrew/include
)

target_link_directories(mlx_inference_engine PUBLIC
    /opt/homebrew/lib
)

find_package(ZLIB REQUIRED)

# Apple Accelerate framework for SIMD optimizations
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

target_link_libraries(mlx_inference_engine
    mlx
    espeak-ng
    sentencepiece
    mecab
    ZLIB::ZLIB
    $<$<PLATFORM_ID:Darwin>:${ACCELERATE_FRAMEWORK}>
)

# Enable espeak-ng fallback for non-Misaki languages (ES, FR, HI, IT, PT-BR)
target_compile_definitions(mlx_inference_engine PUBLIC USE_ESPEAK_FALLBACK)

# Enable MeCab for Japanese kanji tokenization
target_compile_definitions(mlx_inference_engine PUBLIC USE_MECAB)

# Test executable
add_executable(test_mlx_engine
    test_engine.cpp
)

target_link_libraries(test_mlx_engine
    mlx_inference_engine
)

# Phoneme head test executable
add_executable(test_phoneme_head
    test_phoneme_head.cpp
)

target_link_libraries(test_phoneme_head
    mlx_inference_engine
)

# Misaki G2P test executable
add_executable(test_misaki_g2p
    test_misaki_g2p.cpp
)

target_link_libraries(test_misaki_g2p
    mlx_inference_engine
)

# Kokoro benchmark executable
add_executable(benchmark_kokoro
    benchmark_kokoro.cpp
)

target_link_libraries(benchmark_kokoro
    mlx_inference_engine
)

# Misaki G2P benchmark executable
add_executable(benchmark_misaki
    benchmark_misaki.cpp
)

target_link_libraries(benchmark_misaki
    mlx_inference_engine
)

# Whisper STT benchmark executable
add_executable(benchmark_whisper
    benchmark_whisper.cpp
)

target_link_libraries(benchmark_whisper
    mlx_inference_engine
)

# Speech-to-Speech pipeline benchmark executable
add_executable(benchmark_s2s
    benchmark_s2s.cpp
)

target_link_libraries(benchmark_s2s
    mlx_inference_engine
)

# Speech-to-Speech pipeline test executable
add_executable(test_s2s_pipeline
    test_s2s_pipeline.cpp
)

target_link_libraries(test_s2s_pipeline
    mlx_inference_engine
)

# Kokoro profiling executable
add_executable(profile_kokoro
    profile_kokoro.cpp
)

target_link_libraries(profile_kokoro
    mlx_inference_engine
)

# Detailed Kokoro profiling executable
add_executable(profile_kokoro_detailed
    profile_kokoro_detailed.cpp
)

target_link_libraries(profile_kokoro_detailed
    mlx_inference_engine
)

# Prove optimal performance executable
add_executable(prove_optimal
    prove_optimal.cpp
)

target_link_libraries(prove_optimal
    mlx_inference_engine
)

# Prove warmed optimal performance executable
add_executable(prove_warmed_optimal
    prove_warmed_optimal.cpp
)

target_link_libraries(prove_warmed_optimal
    mlx_inference_engine
)

# Enable testing
enable_testing()
add_test(NAME mlx_engine_test COMMAND test_mlx_engine)
