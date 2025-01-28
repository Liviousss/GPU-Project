# Compiler and paths
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Files
TARGET = $(BIN_DIR)/my_program
CU_FILES = $(wildcard $(SRC_DIR)/*.cu)
CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_FILES:.cpp=.o)) $(notdir $(CU_FILES:.cu=.o)))

# Flags
CUDA_ARCH = -arch=sm_75
CXXFLAGS = -std=c++11 -g -I$(CUDA_INCLUDE)
NVCCFLAGS = -std=c++11 $(CUDA_ARCH) -g -G -I$(CUDA_INCLUDE)
LDFLAGS = -L$(CUDA_LIB) -lcudart

# Default target
.PHONY: all
all: build

# Build target
.PHONY: build
build: $(TARGET)

# Compile CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files into the binary
$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ_FILES) -o $@ $(LDFLAGS)

# Run the program
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Clean target
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
