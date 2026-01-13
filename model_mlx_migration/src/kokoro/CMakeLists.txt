cmake_minimum_required(VERSION 3.14)
project(kokoro_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tokenizer Library
add_library(kokoro_tokenizer
    tokenizer.cpp
)

target_include_directories(kokoro_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Prosody Parser Library
add_library(kokoro_prosody
    prosody_parser.cpp
    prosody_adjust.cpp
)

target_include_directories(kokoro_prosody PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test executables
add_executable(test_tokenizer
    test_tokenizer.cpp
)

target_link_libraries(test_tokenizer
    kokoro_tokenizer
)

add_executable(test_prosody
    test_prosody.cpp
)

target_link_libraries(test_prosody
    kokoro_prosody
)

# Enable testing
enable_testing()
add_test(NAME tokenizer_test COMMAND test_tokenizer)
add_test(NAME prosody_test COMMAND test_prosody)
