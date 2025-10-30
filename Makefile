# ========================
#  Makefile for Simplex Solver
# ========================

# --- Directories ---
SRC_DIR  := src
BUILD_DIR := build
BIN_DIR  := bin

# --- Backend selection (default = cuda) ---
BACKEND ?= cuda

# --- Base line uses cuopt, which is an additional library ---
CUOPT_DIR ?= ~/.venv/lib/python3.12/site-packages/libcuopt

# --- Compiler settings ---
CXX      := g++
NVCC     := nvcc
CUOPTCC  := scripts/cuopt_compiler.sh

# --- Compiler flags ---
CXXFLAGS := -std=c++20 -O3
NVCCFLAGS := -std=c++17 -O3
LDFLAGS  := -lcudart -lcusolver

# --- cuOpt configuration
CUOPTFLAGS := $(CXXFLAGS) -I $(CUOPT_DIR)/include/ -L $(CUOPT_DIR)/lib64/
CUOPT_LDFLAGS := -lcuopt -lcusparse

# We need to point LD_LIBRARY_PATH to the cuOpt library.
CUOPT_XENV := LD_LIBRARY_PATH=$(CUOPT_DIR)/lib64/:$$LD_LIBRARY_PATH

ifeq ($(BACKEND),cuda)
  BND_OBJ := $(BUILD_DIR)/solver_gpu.o
  BND_LDFLAGS := -lcublas -lcusolver
  BND_DEP := linalg_gpu
else ifeq ($(BACKEND),cpp)
  BND_OBJ := $(BUILD_DIR)/solver_cpu.o
else ifeq ($(BACKEND),cuopt)
  BND_OBJ := $(BUILD_DIR)/solver_cuopt.o
  BND_LIB := -L $(CUOPT_DIR)/lib64/
  BND_LDFLAGS := $(CUOPT_LDFLAGS)
  BXENV := $(CUOPT_XENV)
else
  $(error Unknown BACKEND '$(BACKEND)'; use BACKEND=cuda or BACKEND=cpp)
endif

# --- Files ---
MAIN_SRC := $(SRC_DIR)/main.cpp
TEST_SRC := $(SRC_DIR)/testing.cpp

DEPS := linprog linalg_cpu logging solver_wrapper

# --- Targets ---
MAIN := $(BIN_DIR)/solver_$(BACKEND)
TEST := $(BIN_DIR)/test_$(BACKEND)
COMPARE := $(BIN_DIR)/compare_$(BACKEND)

# --- Debug build (run `make DEBUG=1`) ---
ifeq ($(DEBUG),1)
  CXXFLAGS += -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
  NVCCFLAGS := -G -lineinfo
endif

# --- Object files ---
MAIN_OBJ := $(BUILD_DIR)/main.o
TEST_OBJ := $(BUILD_DIR)/testing.o
COMPARE_OBJ := $(BUILD_DIR)/compare.o
DEP_OBJ := $(DEPS:%=$(BUILD_DIR)/%.o)
BND_DEP_OBJ := $(BND_DEP:%=$(BUILD_DIR)/%.o)

# ========================
#  Rules
# ========================

.PHONY: all clean run test

all: $(MAIN)

# --- Binaries ---
$(MAIN): $(MAIN_OBJ) $(BND_OBJ) $(BND_DEP_OBJ) $(DEP_OBJ)| $(BIN_DIR)
	$(CXX) $(BND_LIB) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BND_LDFLAGS)

$(TEST): $(TEST_OBJ) $(BND_OBJ) $(BND_DEP_OBJ) $(DEP_OBJ)| $(BIN_DIR)
	$(CXX) $(BND_LIB) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BND_LDFLAGS)

$(COMPARE): $(COMPARE_OBJ) $(BND_OBJ) $(BND_DEP_OBJ) $(DEP_OBJ) $(BUILD_DIR)/solver_cuopt.o | $(BIN_DIR)
	$(CXX) $(BND_LIB) -L $(CUOPT_DIR)/lib64/ $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BND_LDFLAGS) $(CUOPT_LDFLAGS)


# --- Compile rules ---
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/cuopt/%.cpp | $(BUILD_DIR)
	$(CUOPTCC) $(CUOPTFLAGS) -c $< -o $@

# --- Create directories ---
$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

# --- Utility targets ---
run: $(MAIN)
	$(BXENV) ./$(MAIN)

test: $(TEST)
	$(BXENV) ./$(TEST)

compare: $(COMPARE)
	$(CUOPT_XENV) ./$(COMPARE)

clean:
	rm -r $(BUILD_DIR) $(BIN_DIR)
