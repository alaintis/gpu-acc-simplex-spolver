#!/bin/bash

###############################################################################
# Benchmark Pipeline Script
#
# 1. Ask user for backend (gpu / cpu / cuopt)
# 2. Configure CMake with correct backend
# 3. Build benchmark_runner
# 4. Run benchmark_runner across all Netlib presolved files
###############################################################################

DATA_DIR="../../dphpc-simplex-data/netlib/csc/presolved"
BUILD_DIR="build"
EXEC="$BUILD_DIR/benchmark_runner"
TIMEOUT="180s"
RESULTS_DIR="results_selection"
mkdir -p "$RESULTS_DIR"

MAX=64

###############################################################################
# Step 1: Ask user for backend
###############################################################################

echo "Select solver backend:"
echo "  1) GPU usual"
echo "  1b) GPU sherman morrison"
echo "  2) CPU"
echo "  3) multipivot"
echo "  4) cuOpt"
echo "  4) Sparse Hybrid"
echo "  5) GPU v2"
echo -n "Enter choice (1–3): "
read BACKEND_CHOICE

case "$BACKEND_CHOICE" in
    1)
        BACKEND="gpuv0"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=gpuv0"
        ;;
    1b)
        BACKEND="gpuv1"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=gpuv1"
        ;;
    2)
        BACKEND="cpu"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=cpu"
        ;;
    3)
        BACKEND="gpu_multi"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=gpu_multi"
        ;;
    4)
        BACKEND="cuopt"
        echo -n "Enter path to cuOpt_DIR (or press Enter to skip): "
        read CUOPT_PATH
        if [ -n "$CUOPT_PATH" ]; then
            CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=cuopt -DcuOpt_DIR=$CUOPT_PATH"
        else
            CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=cuopt"
        fi
        ;;
    4)
        BACKEND="sparse"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=sparse"
        ;;
    5)
        BACKEND="gpu-multi"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=gpu-multi"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Using backend: $BACKEND"
echo "Running CMake:"
echo "  $CMAKE_CMD"
echo ""
OUTFILE="$RESULTS_DIR/bench_results_${TIMEOUT}_${BACKEND}.txt"
LOGFILE="$RESULTS_DIR/bench_full_log_${TIMEOUT}_${BACKEND}.txt"

###############################################################################
# Step 2: Configure CMake
###############################################################################

$CMAKE_CMD
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Exiting."
    exit 1
fi

###############################################################################
# Step 3: Build benchmark_runner
###############################################################################

echo ""
echo "Building benchmark_runner..."
cmake --build "$BUILD_DIR" --target benchmark_runner
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

if [ ! -f "$EXEC" ]; then
    echo "benchmark_runner executable not found! Exiting."
    exit 1
fi

chmod +x "$EXEC"

###############################################################################
# Step 4: Run Benchmark Batch
###############################################################################

echo "Cleaning output logs..."
> "$OUTFILE"
> "$LOGFILE"

echo "----------------------------------------------"
echo "IMPORTANT: PLEASE READ CAREFULLY"
echo "We are benchmarking the netlib presolved problems"
echo "Running benchmarks on backend: $BACKEND"
echo "Results will be saved to: $OUTFILE"
echo "time limit per problem: $TIMEOUT"
echo "---------------------------------------------"

count=0

for file in "$DATA_DIR"/*.csc; do
    ((count++))
    if [ $count -gt $MAX ]; then
        echo "Stopping early after $MAX problems."
        break
    fi

    echo "Running file #$count: $file"

    # use srun to match your working CLI behavior
    OUTPUT=$(timeout $TIMEOUT srun -A dphpc -t 00:10 "$EXEC" "$file" 2>&1)
    EXITCODE=$?

    {
        echo "============ $file ============"
        echo "$OUTPUT"
        echo ""
    } >> "$LOGFILE"

    if [ $EXITCODE -eq 124 ]; then
        echo "  -> TIMEOUT"
        continue
    fi

    if [ $EXITCODE -ne 0 ]; then
        echo "  -> CRASH (exit code $EXITCODE)"
        continue
    fi

    BLOCK=$(echo "$OUTPUT" | sed -n '/^START$/,/^END$/p')
    if [ -z "$BLOCK" ]; then
        echo "  -> FAILED (no START/END block)"
        continue
    fi

    CLEANED=$(echo "$BLOCK" | sed '/^START$/d;/^END$/d')

    echo "$file" >> "$OUTFILE"
    echo "$CLEANED" >> "$OUTFILE"
    echo "" >> "$OUTFILE"

    echo "  -> OK"
done

echo "---------------------------------------------"
echo "Done. Results in: $OUTFILE"
echo "Full logs in:    $LOGFILE"
echo "Problems run:    $count"
