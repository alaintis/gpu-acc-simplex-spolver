#!/bin/bash

################################################################################
# Simplex Solver - Batch Test Runner (Interactive Backend Version)
################################################################################
#
# 1. Ask user for backend (gpu / cpu / cuopt)
# 2. Configure CMake accordingly
# 3. Build test_runner
# 4. Run test_runner across all Netlib presolved problems
################################################################################

DATA_DIR="../../dphpc-simplex-data/netlib/csc/presolved"
BUILD_DIR="build"
EXEC="$BUILD_DIR/testing_execution"
TIMEOUT="100s"
RESULTS_DIR="results_selection"
mkdir -p "$RESULTS_DIR"


MAX=80

################################################################################
# Step 1: Ask for backend
################################################################################

echo "Select solver backend to build:"
echo "  1) GPU usual"
echo "  1b) GPU sherman morrison, type: 1b"
echo "  2) gpu_multi"
echo "  3) CPU"
echo "  4) cuOpt"
echo "  5) Sparse Hybrid"
echo "  6) GPU v2"
echo -n "Enter choice (1–6): "
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
        BACKEND="gpu_multi"
        CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=gpu_multi"
        ;;
    3)
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

OUTFILE="$RESULTS_DIR/time_testing_results_${TIMEOUT}_${BACKEND}.txt"
LOGFILE="$RESULTS_DIR/full_log_${TIMEOUT}_${BACKEND}.txt"
echo ""
echo "Using backend: $BACKEND"
echo "Running CMake:"
echo "  $CMAKE_CMD"
echo ""

################################################################################
# Step 2: CMake configure
################################################################################

$CMAKE_CMD
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Exiting."
    exit 1
fi

################################################################################
# Step 3: Build test_runner
################################################################################

echo ""
echo "Building test_runner..."
cmake --build "$BUILD_DIR" --target testing_execution
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

if [ ! -f "$EXEC" ]; then
    echo "ERROR: test_runner not found at $EXEC"
    exit 1
fi

chmod +x "$EXEC"

################################################################################
# Step 4: Original batch-execution logic
################################################################################

> "$OUTFILE"
> "$LOGFILE"

count=0

echo "----------------------------------------------"
echo "IMPORTANT: PLEASE READ CAREFULLY"
echo "We are testing the netlib presolved problems"
echo "Running tests on backend: $BACKEND"
echo "Results will be saved to: $OUTFILE"
echo "time limit per problem: $TIMEOUT"
echo "---------------------------------------------"


for file in "$DATA_DIR"/*.csc; do
    ((count++))

    if [ $count -gt $MAX ]; then
        echo "Stopping early after $MAX problems."
        break
    fi

    echo "Running file #$count: $file"

    OUTPUT=$(timeout $TIMEOUT srun -A dphpc -t 00:10 "$EXEC" "$file" 2>&1)
    EXITCODE=$?

    {
        echo "============ $file ============"
        echo "$OUTPUT"
        echo ""
    } >> "$LOGFILE"

    if [ $EXITCODE -eq 124 ]; then
        echo "  -> TIMEOUT ($TIMEOUT)"
        continue
    fi

    if [ $EXITCODE -ne 0 ]; then
        echo "  -> CRASH (exit code $EXITCODE)"
        continue
    fi

    BLOCK=$(echo "$OUTPUT" | sed -n '/^START$/,/^END$/p')
    if [[ -z "$BLOCK" ]]; then
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
echo "Completed. Valid results saved to: $OUTFILE"
echo "Full logs saved to:               $LOGFILE"
echo "Total files processed:            $count"
