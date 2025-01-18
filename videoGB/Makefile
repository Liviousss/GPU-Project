# Compiler Setup
CC = g++
NVCC = nvcc
CXXFLAGS = -O3 -Wall
CUDAFLAGS = -arch=sm_75  # Adjust for your GPU architecture
LDFLAGS = -lcudart  # Link to CUDA runtime

# Directory Setup
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source and Object Files
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)  # All .cpp files in src/
CU_SOURCE = $(SRC_DIR)/cuda_kernel.cu  # CUDA source file in the src folder
OBJ_FILES = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) $(OBJ_DIR)/kernel.o  # Object files

# Executable name
EXEC = $(BIN_DIR)/my_program

# Default target to build the program
all: $(EXEC)

# Compile .cpp files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

# Compile .cu file
$(OBJ_DIR)/kernel.o: $(CU_SOURCE)
	$(NVCC) -c $(CU_SOURCE) -o $(OBJ_DIR)/kernel.o $(CUDAFLAGS)

# Link object files to create the executable
$(EXEC): $(OBJ_FILES)
	$(NVCC) $(OBJ_FILES) -o $(EXEC) $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -rf $(OBJ_DIR)/*.o $(EXEC)

# Create the necessary directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@