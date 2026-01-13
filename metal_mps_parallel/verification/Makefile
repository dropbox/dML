# Makefile for platform assumption verification tools

CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
OBJCFLAGS = -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation

# Include path for header
INCLUDES = -I.

# Source files
SOURCES = runtime_assumptions.mm run_platform_checks.mm

# Output binary
TARGET = run_platform_checks

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) runtime_assumptions.h
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $(INCLUDES) $(FRAMEWORKS) \
		runtime_assumptions.mm run_platform_checks.mm \
		-o $(TARGET)
	@echo "Built $(TARGET)"

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# Generate reports
report-json: $(TARGET)
	./$(TARGET) --json > platform_report.json
	@echo "Wrote platform_report.json"

report-md: $(TARGET)
	./$(TARGET) --markdown > platform_report.md
	@echo "Wrote platform_report.md"

reports: report-json report-md
