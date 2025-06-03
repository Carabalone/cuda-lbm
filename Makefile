# Compiler and build configuration
CC = nvcc

# Build mode: release or debug (default: release)
BUILD ?= release

# Flag components
COMMON_CFLAGS = \
    -arch=sm_61 \
    -O3 \
    -rdc=true \
    -Isrc/ \
    -Ithird_party/ \
    --std=c++17 \
    --compiler-options -Wall,-Wextra,-std=c++17

CFLAGS_release = $(COMMON_CFLAGS)
CFLAGS_debug   = $(COMMON_CFLAGS) -g -G -lineinfo

CFLAGS  = $(CFLAGS_$(BUILD))
LDFLAGS = $(CFLAGS) -lcudart

# Directories
SRCDIR    = src
BINDIR    = bin
OUTPUTDIR = output
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
.PHONY: release debug
release: BUILD = release
release: all

debug: BUILD = debug
debug: all

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