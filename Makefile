# Compiler and flags
CC = nvcc
CFLAGS = -arch=sm_61 -O3 -rdc=true -g -G -Isrc/ --std=c++17 --compiler-options -Wall,-Wextra,-std=c++17
LDFLAGS = $(CFLAGS) -lcudart

# Directories
SRCDIR = src
BINDIR = bin
OUTPUTDIR = output
TARGET = $(BINDIR)/lbm_solver

# Source files
CXX_SRCS = $(shell find $(SRCDIR) -name "*.cpp")
CU_SRCS = $(shell find $(SRCDIR) -name "*.cu")
CXX_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.cpp.o,$(CXX_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%.cu.o,$(CU_SRCS))
OBJS = $(CXX_OBJS) $(CU_OBJS)

# Dependency files
DEPS = $(OBJS:.o=.d)

# Default target
all: $(TARGET)

# Include dependencies
-include $(DEPS)

# Link target
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile C++ files
$(BINDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MD -MF $(@:.o=.d) -c $< -o $@

# Compile CUDA files
$(BINDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MD -MF $(@:.o=.d) -c $< -o $@

# Create binary directory if needed
$(OBJS): | $(BINDIR)

$(BINDIR):
	mkdir -p $@

# Run the program
run: $(TARGET)
	@mkdir -p $(OUTPUTDIR)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BINDIR)/* $(OUTPUTDIR)/*

# View preprocessed code
preprocess: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -E $< -o $(OUTPUTDIR)/main_preprocessed.cu

# View PTX code
ptx: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -ptx $< -o $(OUTPUTDIR)/main.ptx

# View CUBIN code
cubin: $(SRCDIR)/main.cu
	$(CC) $(CFLAGS) -cubin $< -o $(OUTPUTDIR)/main.cubin

.PHONY: all run clean preprocess ptx cubin
