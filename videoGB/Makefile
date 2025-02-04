# Compiler
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Executable
TARGET = $(BIN_DIR)/my_program

# CUDA Architecture
ARCH = -arch=sm_75

# Compiler Flags
CXXFLAGS = -std=c++11 -g -fPIC
NVCCFLAGS = $(ARCH) -std=c++11 -g --compiler-options '-fPIC'

# Include & Library Paths
INCLUDES = -I/usr/local/cuda/include -I/usr/include/opencv4
LIBS = -L/usr/local/cuda/lib64 -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

# Find all source files
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Generate object file paths
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))

# Create obj and bin directories if they don’t exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Link all objects and create executable
$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPP_OBJECTS) $(CU_OBJECTS) $(LIBS) -o $(TARGET)

# Build the project
build: $(TARGET)
	@echo "Build complete."

# Run the compiled binary
run: $(TARGET)
	@echo "Running..."
	./$(TARGET)

# Clean object files and binary
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned up build files."


.PHONY: build run clean
