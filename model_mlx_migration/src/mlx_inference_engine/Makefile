# MLX Inference Engine - Makefile
# Builds the unified C++ inference engine

CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I. -I../kokoro -I/opt/homebrew/include -I/opt/homebrew/Cellar/sentencepiece/0.2.1/include -I$(HOME)/.local/include
LDFLAGS = -L/opt/homebrew/lib -L/opt/homebrew/Cellar/sentencepiece/0.2.1/lib -L$(HOME)/.local/lib -Wl,-rpath,$(HOME)/.local/lib
LIBS = -lmlx -lespeak-ng -lsentencepiece -lz

# Source files
ENGINE_SRCS = mlx_inference_engine.cpp translation_model.cpp whisper_model.cpp llm_model.cpp silero_vad.cpp output_writers.cpp grammar_parser.cpp cosyvoice3_model.cpp
KOKORO_SRCS = ../kokoro/kokoro.cpp ../kokoro/model.cpp ../kokoro/g2p.cpp ../kokoro/tokenizer.cpp
TEST_SRCS = test_engine.cpp

# Object files
ENGINE_OBJS = mlx_inference_engine.o translation_model.o whisper_model.o llm_model.o silero_vad.o output_writers.o grammar_parser.o cosyvoice3_model.o
KOKORO_OBJS = kokoro.o model.o g2p.o tokenizer.o prosody_parser.o prosody_adjust.o

# Targets
.PHONY: all clean test

all: test_engine

# Compile Kokoro components (from parent directory)
kokoro.o: ../kokoro/kokoro.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

model.o: ../kokoro/model.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

g2p.o: ../kokoro/g2p.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tokenizer.o: ../kokoro/tokenizer.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

prosody_parser.o: ../kokoro/prosody_parser.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

prosody_adjust.o: ../kokoro/prosody_adjust.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile engine
mlx_inference_engine.o: mlx_inference_engine.cpp mlx_inference_engine.hpp translation_model.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile translation model
translation_model.o: translation_model.cpp translation_model.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile whisper model
whisper_model.o: whisper_model.cpp whisper_model.h silero_vad.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile Silero VAD
silero_vad.o: silero_vad.cpp silero_vad.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile LLM model
llm_model.o: llm_model.cpp llm_model.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile output writers
output_writers.o: output_writers.cpp output_writers.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile grammar parser
grammar_parser.o: grammar_parser.cpp grammar_parser.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CosyVoice3 model
cosyvoice3_model.o: cosyvoice3_model.cpp cosyvoice3_model.h llm_model.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile test
test_engine.o: test_engine.cpp mlx_inference_engine.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link test executable
test_engine: test_engine.o $(ENGINE_OBJS) $(KOKORO_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@

# Run tests (without model - just tests error handling)
test: test_engine
	./test_engine

# Run tests with Kokoro model
test_kokoro: test_engine
	./test_engine ~/models/kokoro/kokoro_cpp_export

# Clean
clean:
	rm -f *.o test_engine mlx_engine_output.wav

# Help
help:
	@echo "MLX Inference Engine Build"
	@echo "=========================="
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build test_engine (default)"
	@echo "  test       - Run tests (error handling only)"
	@echo "  test_kokoro - Run tests with Kokoro model"
	@echo "  clean      - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build"
	@echo "  make test         # Run basic tests"
	@echo "  make test_kokoro  # Run with Kokoro model"
