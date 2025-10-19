# ========================
#  Makefile for Simplex Solver
# ========================

# --- Directories ---
SRC_DIR  := src
BUILD_DIR := build
BIN_DIR  := bin

CPP_BACKEND_SRC  := $(SRC_DIR)/solver_cpu.cpp
CUDA_BACKEND_SRC := $(SRC_DIR)/solver_gpu.cu

# --- Backend selection (default = cuda) ---
BACKEND ?= cuda

# --- Compiler settings ---
CXX      := g++
NVCC     := nvcc

ifeq ($(BACKEND),cuda)
  BND_SRC := $(CUDA_BACKEND_SRC)
  BND_OBJ := $(BUILD_DIR)/solver_gpu.o
  CBND := $(NVCC)
  BNDFLAGS := -O3 -std=c++17
else ifeq ($(BACKEND),cpp)
  BND_SRC := $(CPP_BACKEND_SRC)
  BND_OBJ := $(BUILD_DIR)/solver_cpu.o
  CBND := $(CXX)
  BNDFLAGS := -O3 -std=c++20
  BND_DEP := linalg
else
  $(error Unknown BACKEND '$(BACKEND)'; use BACKEND=cuda or BACKEND=cpp)
endif

# --- Compiler flags ---
CXXFLAGS := -std=c++20 -O3
LDFLAGS  := -lcudart -lcusolver

# --- Files ---
MAIN_SRC := $(SRC_DIR)/main.cpp
TEST_SRC := $(SRC_DIR)/testing.cpp

DEPS := linprog

# --- Targets ---
MAIN := $(BIN_DIR)/solver_$(BACKEND)
TEST := $(BIN_DIR)/test_$(BACKEND)

# --- Debug build (run `make DEBUG=1`) ---
ifeq ($(DEBUG),1)
  CXXFLAGS += -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
  NVCCFLAGS := -G -lineinfo
endif

# --- Object files ---
MAIN_OBJ := $(BUILD_DIR)/main.o
TEST_OBJ := $(BUILD_DIR)/testing.o
DEP_OBJ := $(DEPS:%=$(BUILD_DIR)/%.o)
BND_DEP_OBJ := $(BND_DEP:%=$(BUILD_DIR)/%.o)

# ========================
#  Rules
# ========================

.PHONY: all clean run test

all: $(MAIN)

# --- Binaries ---
$(MAIN): $(MAIN_OBJ) $(BND_OBJ) $(BND_DEP_OBJ) $(DEP_OBJ)| $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST): $(TEST_OBJ) $(BND_OBJ) $(BND_DEP_OBJ) $(DEP_OBJ)| $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Compile rules ---
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BND_OBJ): $(BND_SRC) | $(BUILD_DIR)
	$(CBND) $(BNDFLAGS) -c $< -o $@

# --- Create directories ---
$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

# --- Utility targets ---
run: $(MAIN)
	./$(MAIN)

test: $(TEST)
	./$(TEST)

clean:
	rm -r $(BUILD_DIR) $(BIN_DIR)
