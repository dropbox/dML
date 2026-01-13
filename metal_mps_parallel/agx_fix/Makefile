# AGX Driver Fix Library
# Builds libagx_fix.dylib for injection via DYLD_INSERT_LIBRARIES

CC = clang++
CFLAGS = -std=c++17 -O2 -Wall -Wextra -fPIC
FRAMEWORKS = -framework Foundation -framework Metal
FRAMEWORKS_MPS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
LDFLAGS_BASE = -dynamiclib
LDFLAGS = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix.dylib
LDFLAGS_OPT = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_optimized.dylib

SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

SOURCES = $(SRC_DIR)/agx_fix.mm
SOURCES_OPT = $(SRC_DIR)/agx_fix_optimized.mm
SOURCES_COMP = $(SRC_DIR)/agx_fix_comprehensive.mm
SOURCES_V2 = $(SRC_DIR)/agx_fix_v2.mm
SOURCES_V2_2 = $(SRC_DIR)/agx_fix_v2_2.mm
SOURCES_V2_3 = $(SRC_DIR)/agx_fix_v2_3.mm
SOURCES_V2_4 = $(SRC_DIR)/agx_fix_v2_4.mm
SOURCES_V2_4_NR = $(SRC_DIR)/agx_fix_v2_4_neverrelease.mm
SOURCES_V2_5 = $(SRC_DIR)/agx_fix_v2_5.mm
SOURCES_V2_6 = $(SRC_DIR)/agx_fix_v2_6.mm
SOURCES_V2_7 = $(SRC_DIR)/agx_fix_v2_7.mm
SOURCES_V2_8 = $(SRC_DIR)/agx_fix_v2_8.mm
SOURCES_V2_9 = $(SRC_DIR)/agx_fix_v2_9.mm
LIBRARY = $(BUILD_DIR)/libagx_fix.dylib
LIBRARY_OPT = $(BUILD_DIR)/libagx_fix_optimized.dylib
LIBRARY_COMP = $(BUILD_DIR)/libagx_fix_comprehensive.dylib
LIBRARY_V2 = $(BUILD_DIR)/libagx_fix_v2.dylib
LIBRARY_V2_2 = $(BUILD_DIR)/libagx_fix_v2_2.dylib
LIBRARY_V2_3 = $(BUILD_DIR)/libagx_fix_v2_3.dylib
LIBRARY_V2_4 = $(BUILD_DIR)/libagx_fix_v2_4.dylib
LIBRARY_V2_4_NR = $(BUILD_DIR)/libagx_fix_v2_4_nr.dylib
LIBRARY_V2_5 = $(BUILD_DIR)/libagx_fix_v2_5.dylib
LIBRARY_V2_6 = $(BUILD_DIR)/libagx_fix_v2_6.dylib
LIBRARY_V2_7 = $(BUILD_DIR)/libagx_fix_v2_7.dylib
LIBRARY_V2_8 = $(BUILD_DIR)/libagx_fix_v2_8.dylib
LIBRARY_V2_9 = $(BUILD_DIR)/libagx_fix_v2_9.dylib
LDFLAGS_COMP = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_comprehensive.dylib
LDFLAGS_V2 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2.dylib
LDFLAGS_V2_2 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_2.dylib
LDFLAGS_V2_3 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_3.dylib
LDFLAGS_V2_4 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_4.dylib
LDFLAGS_V2_4_NR = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_4_nr.dylib
LDFLAGS_V2_5 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_5.dylib
LDFLAGS_V2_6 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_6.dylib
LDFLAGS_V2_7 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_7.dylib
LDFLAGS_V2_8 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_8.dylib
LDFLAGS_V2_9 = $(LDFLAGS_BASE) -install_name @rpath/libagx_fix_v2_9.dylib

.PHONY: all clean install test test-optimized \
	v2 v2_2 v2_3 v2_4 v2_4_nr v2_5 v2_6 v2_7 v2_8 v2_9

all: $(LIBRARY) $(LIBRARY_OPT) $(LIBRARY_COMP) $(LIBRARY_V2) $(LIBRARY_V2_2) $(LIBRARY_V2_3) $(LIBRARY_V2_4) $(LIBRARY_V2_4_NR) $(LIBRARY_V2_5) $(LIBRARY_V2_6) $(LIBRARY_V2_7) $(LIBRARY_V2_8) $(LIBRARY_V2_9)

v2: $(LIBRARY_V2)
v2_2: $(LIBRARY_V2_2)
v2_3: $(LIBRARY_V2_3)
v2_4: $(LIBRARY_V2_4)
v2_4_nr: $(LIBRARY_V2_4_NR)
v2_5: $(LIBRARY_V2_5)
v2_6: $(LIBRARY_V2_6)
v2_7: $(LIBRARY_V2_7)
v2_8: $(LIBRARY_V2_8)
v2_9: $(LIBRARY_V2_9)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIBRARY): $(SOURCES) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_OPT): $(SOURCES_OPT) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_OPT) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_COMP): $(SOURCES_COMP) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_COMP) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2): $(SOURCES_V2) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_2): $(SOURCES_V2_2) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2_2) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_3): $(SOURCES_V2_3) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2_3) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_4): $(SOURCES_V2_4) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2_4) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_4_NR): $(SOURCES_V2_4_NR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2_4_NR) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_5): $(SOURCES_V2_5) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS_MPS) $(LDFLAGS_V2_5) -o $@ $<
	@echo "Built: $@ (with MetalPerformanceShaders support)"

$(LIBRARY_V2_6): $(SOURCES_V2_6) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS_V2_6) -o $@ $<
	@echo "Built: $@"

$(LIBRARY_V2_7): $(SOURCES_V2_7) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS_MPS) $(LDFLAGS_V2_7) -o $@ $<
	@echo "Built: $@ (with commit-safety for validation fix)"

$(LIBRARY_V2_8): $(SOURCES_V2_8) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS_MPS) $(LDFLAGS_V2_8) -o $@ $<
	@echo "Built: $@ (with event-safety for Bug #048 fix)"

$(LIBRARY_V2_9): $(SOURCES_V2_9) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FRAMEWORKS_MPS) $(LDFLAGS_V2_9) -o $@ $<
	@echo "Built: $@ (with formal verification gap fixes)"
	@echo ""
	@echo "AGX Fix Library Variants:"
	@echo "  v2.9 (gap fixes):           DYLD_INSERT_LIBRARIES=$(LIBRARY_V2_9) your_app  [RECOMMENDED]"
	@echo "  v2.8 (event-safety):        DYLD_INSERT_LIBRARIES=$(LIBRARY_V2_8) your_app"
	@echo "  v2.7 (commit-safety):       DYLD_INSERT_LIBRARIES=$(LIBRARY_V2_7) your_app"
	@echo ""
	@echo "v2.9 FORMAL VERIFICATION GAP FIXES:"
	@echo "  GAP 1: Mutex held through entire commit (closes race window)"
	@echo "  GAP 2: Block indefinitely on event ops (no timeout escape)"
	@echo "  GAP 3: parallelRenderCommandEncoder tracked"
	@echo ""
	@echo "Environment variables:"
	@echo "  AGX_FIX_DISABLE=1  - Disable the fix"
	@echo "  AGX_FIX_VERBOSE=1  - Enable verbose logging"
	@echo "  AGX_FIX_DEADLOCK_DETECT=1  - Enable mutex wait/deadlock diagnostics"
	@echo "    AGX_FIX_LOCK_WARN_MS=1000          - First warning threshold (ms)"
	@echo "    AGX_FIX_LOCK_LOG_INTERVAL_MS=5000  - Warning repeat interval (ms)"
	@echo "    AGX_FIX_LOCK_TIMEOUT_MS=0          - Timeout threshold (ms, 0=disabled)"
	@echo "    AGX_FIX_LOCK_ABORT_ON_TIMEOUT=1    - Abort the process on timeout"

clean:
	rm -rf $(BUILD_DIR)

# Run test to verify the fix works
test: $(LIBRARY)
	@echo "Running test with global mutex fix..."
	DYLD_INSERT_LIBRARIES=$(LIBRARY) AGX_FIX_VERBOSE=1 python3 -c "import torch; print('PyTorch loaded successfully with AGX fix')"
	@echo ""
	@echo "Running multi-threaded stress test..."
	DYLD_INSERT_LIBRARIES=$(LIBRARY) python3 $(TEST_DIR)/test_agx_fix.py

# Run test with optimized per-encoder mutex fix
test-optimized: $(LIBRARY_OPT)
	@echo "Running test with per-encoder mutex fix (OPTIMIZED)..."
	DYLD_INSERT_LIBRARIES=$(LIBRARY_OPT) AGX_FIX_VERBOSE=1 python3 -c "import torch; print('PyTorch loaded successfully with AGX fix (optimized)')"
	@echo ""
	@echo "Running multi-threaded stress test with optimized fix..."
	DYLD_INSERT_LIBRARIES=$(LIBRARY_OPT) python3 $(TEST_DIR)/test_agx_fix.py

# Install to /usr/local/lib (requires sudo)
install: $(LIBRARY)
	sudo cp $(LIBRARY) /usr/local/lib/
	@echo "Installed to /usr/local/lib/libagx_fix.dylib"
	@echo ""
	@echo "Usage: DYLD_INSERT_LIBRARIES=/usr/local/lib/libagx_fix.dylib your_app"
