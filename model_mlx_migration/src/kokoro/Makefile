# Kokoro C++ TTS Makefile
# Optimized for Apple Silicon (ARM NEON)

CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra

# Include paths
INCLUDES = -I/opt/homebrew/include

# Library paths and libraries
LDFLAGS = -L/opt/homebrew/lib
LIBS = -lmlx -lespeak-ng

# Source files
COMMON_SRCS = kokoro.cpp model.cpp g2p.cpp tokenizer.cpp prosody_parser.cpp prosody_adjust.cpp

# Optimization levels
# Debug build
DEBUG_FLAGS = -g -O0 -DDEBUG

# Release build (standard)
RELEASE_FLAGS = -O2 -DNDEBUG

# Optimized release build (maximum performance)
# -O3: Maximum optimization
# -march=native: Use all CPU features available
# -mtune=native: Tune for current CPU
# -ffast-math: Allow aggressive floating-point optimizations
# -flto: Link-time optimization
# -funroll-loops: Unroll loops for better performance
OPTIMIZED_FLAGS = -O3 -DNDEBUG -march=native -mtune=native -ffast-math -flto -funroll-loops

# Default to optimized build
BUILD_FLAGS ?= $(OPTIMIZED_FLAGS)

# Targets
.PHONY: all clean debug release optimized test benchmark

all: test_pipeline test_token_input test_voice_switching test_language_switching

# Build individual executables
test_pipeline: test_pipeline.cpp $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LIBS) -o $@

test_token_input: test_token_input.cpp $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LIBS) -o $@

test_voice_switching: test_voice_switching.cpp $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LIBS) -o $@

test_language_switching: test_language_switching.cpp $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LIBS) -o $@

test_model: test_model.cpp $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LIBS) -o $@

test_mlx_load: test_mlx_load.cpp model.cpp
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) -lmlx -o $@

# Build modes
debug:
	$(MAKE) BUILD_FLAGS="$(DEBUG_FLAGS)" all

release:
	$(MAKE) BUILD_FLAGS="$(RELEASE_FLAGS)" all

optimized:
	$(MAKE) BUILD_FLAGS="$(OPTIMIZED_FLAGS)" all

# Tokenizer tests (no MLX dependency)
test_tokenizer: test_tokenizer.cpp tokenizer.cpp
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $^ -o $@

test_g2p: test_g2p.cpp g2p.cpp
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) -lespeak-ng -o $@

test_integration: test_integration.cpp tokenizer.cpp g2p.cpp
	$(CXX) $(CXXFLAGS) $(BUILD_FLAGS) $(INCLUDES) $^ $(LDFLAGS) -lespeak-ng -o $@

# Run tests
test: test_tokenizer test_g2p test_integration
	./test_tokenizer
	./test_g2p
	./test_integration

# Benchmark (requires model path)
benchmark: optimized
	@echo "Running benchmark with optimized build..."
	@echo "Usage: MODEL_PATH=/path/to/model make benchmark"
	@if [ -n "$(MODEL_PATH)" ]; then \
		for i in 1 2 3 4 5; do \
			./test_pipeline $(MODEL_PATH) "Hello world, this is a test of the kokoro text to speech system."; \
		done; \
	else \
		echo "Set MODEL_PATH to run benchmark"; \
	fi

# Clean build artifacts
clean:
	rm -f test_pipeline test_token_input test_voice_switching test_language_switching
	rm -f test_model test_mlx_load test_tokenizer test_g2p test_integration
	rm -f *.o *.wav

# Show build info
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS) $(BUILD_FLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Libraries: $(LIBS)"
	@echo ""
	@echo "Build targets:"
	@echo "  make           - Build all with optimized flags"
	@echo "  make debug     - Build with debug flags (-g -O0)"
	@echo "  make release   - Build with release flags (-O2)"
	@echo "  make optimized - Build with maximum optimization (-O3 -march=native)"
	@echo "  make test      - Build and run unit tests"
	@echo "  make benchmark - Run performance benchmark"
	@echo "  make clean     - Remove build artifacts"
