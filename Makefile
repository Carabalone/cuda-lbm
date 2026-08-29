# Compiler and build configuration
CC = nvcc

# Build mode: release or debug (default: release)
BUILD ?= release

# Auto-detect CUDA arch from primary GPU (1050 Ti -> sm_61, 2080 -> sm_75).
# Override: make GPU_ARCH=sm_75   or   CUDA_ARCH=sm_75 make
GPU_ARCH ?= $(shell bash scripts/detect_cuda_arch.sh)
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$$//')

# Flag components
COMMON_CFLAGS = \
    -arch=$(GPU_ARCH) \
    -O3 \
    -rdc=true \
    -Isrc/ \
    -Ithird_party/ \
    --std=c++17 \
    --compiler-options -Wall,-Wextra,-std=c++17

CFLAGS_release = $(COMMON_CFLAGS)
CFLAGS_debug   = $(COMMON_CFLAGS) -g -G -lineinfo

CFLAGS  = $(CFLAGS_$(BUILD)) $(EXTRA_DEFINES)
LDFLAGS = $(CFLAGS) -lcudart

# Directories
SRCDIR    = src
BINDIR    ?= bin
OUTPUTDIR = output
FIGDIR    = figures
TARGET    = $(BINDIR)/lbm_solver

# Source files
CXX_SRCS = $(shell find $(SRCDIR) -name "*.cpp")
CU_SRCS  = $(shell find $(SRCDIR) -name "*.cu")

CXX_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.cpp.o,$(CXX_SRCS))
CU_OBJS  = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%.cu.o,$(CU_SRCS))
OBJS     = $(CXX_OBJS) $(CU_OBJS)

# Dependency files
DEPS = $(OBJS:.o=.d)

# Default target: release build
all: $(TARGET)

# Phony aliases to switch modes
.PHONY: release debug show-gpu bench-matrix bench-matrix-196 bench-plot bench-all
release: BUILD = release
release: all

debug: BUILD = debug
debug: all

show-gpu:
	@echo "GPU:      $(GPU_NAME)"
	@echo "CUDA arch: $(GPU_ARCH)  (-arch=$(GPU_ARCH))"

# Full TGV perf matrix (128^3). Writes benchmark_results_<gpu-slug>_128.txt
bench-matrix:
	@./scripts/run_benchmark_matrix.sh

# Optional 196^3 grid (whitepaper Figure 5.9)
bench-matrix-196:
	@GRID=196 ./scripts/run_benchmark_matrix.sh

# Regenerate figures/fig_5_*.png from available CSV files
bench-plot:
	@python3 scripts/plot_perf_figures.py || (echo "Need matplotlib: pip install --user matplotlib" && exit 1)

bench-all: bench-matrix bench-plot

# Include automatically-generated dependency files
-include $(DEPS)

# Link
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile C++ sources
$(BINDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MD -MF $(@:.o=.d) -c $< -o $@

# Compile CUDA sources
$(BINDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MD -MF $(@:.o=.d) -c $< -o $@

# Ensure binary dir exists
$(OBJS): | $(BINDIR)

$(BINDIR):
	mkdir -p $@

# Run, clean, and inspection targets
.PHONY: run clean preprocess ptx cubin
run: $(TARGET)
	@mkdir -p $(OUTPUTDIR)
	./$(TARGET)

clean:
	rm -rf $(BINDIR)/* $(OUTPUTDIR)/*

preprocess: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -E $< -o $(OUTPUTDIR)/main_preprocessed.cu

ptx: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -ptx $< -o $(OUTPUTDIR)/main.ptx

cubin: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -cubin $< -o $(OUTPUTDIR)/main.cubin
