#!/bin/bash

################################################################################
# Compare Tool Runner (Backend-aware)
################################################################################
#
# 1. Ask backend (gpu/cpu/cuopt)
# 2. Configure cmake
# 3. Build compare_tool
# 4. Run compare_tool via SLURM
# 5. Extract data between START/END blocks
################################################################################

BUILD_DIR="build"
EXEC="$BUILD_DIR/compare_tool"


################################################################################
# Backend selection
################################################################################

echo "Select solver backend to build:"
echo "  1) GPU usual"
echo "  1b) GPU sherman morrison"
echo "  2) CPU"
echo "  3) cuOpt"

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
        BACKEND="cuopt"
        echo -n "Enter cuOpt_DIR path (Enter=none): "
        read CUOPT_PATH
        if [ -n "$CUOPT_PATH" ]; then
            CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=cuopt -DcuOpt_DIR=$CUOPT_PATH"
        else
            CMAKE_CMD="cmake -S . -B $BUILD_DIR -DSOLVER_BACKEND=cuopt"
        fi
        ;;
    *)
        echo "Invalid selection."
        exit 1
        ;;
esac

echo ""
echo "Running CMake configuration:"
echo "$CMAKE_CMD"
echo ""

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
OUTFILE="$RESULTS_DIR/compare_results_${BACKEND}.txt"
LOGFILE="$RESULTS_DIR/compare_full_log_${BACKEND}.txt"

# Clean old outputs
> "$OUTFILE"
> "$LOGFILE"

################################################################################
# Configure CMake
################################################################################
$CMAKE_CMD
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

################################################################################
# Build compare_tool
################################################################################

echo "Building compare_tool..."
cmake --build "$BUILD_DIR" --target compare_tool
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

if [ ! -f "$EXEC" ]; then
    echo "compare_tool not found! Exiting."
    exit 1
fi

chmod +x "$EXEC"

################################################################################
# Run compare_tool
################################################################################

echo ""
echo "Running compare_tool via SLURM..."
echo ""

OUTPUT=$(srun -A dphpc -t 00:10 "$EXEC" 2>&1)
EXITCODE=$?

# Log everything
{
    echo "================ FULL OUTPUT ==============="
    echo "$OUTPUT"
    echo ""
} >> "$LOGFILE"

if [ $EXITCODE -ne 0 ]; then
    echo "Execution failed (exit code $EXITCODE)."
    exit 1
fi

################################################################################
# Extract all START/END blocks
################################################################################

echo "Extracting data blocks..."

# Count occurrences
BLOCK_COUNT=$(echo "$OUTPUT" | grep -c "^START$")

if [ "$BLOCK_COUNT" -eq 0 ]; then
    echo "No START/END blocks found."
    exit 1
fi

echo "Found $BLOCK_COUNT data blocks."

echo "" >> "$OUTFILE"
echo "===== Extracted Comparison Data =====" >> "$OUTFILE"

# Extract blocks
echo "$OUTPUT" | awk '
    /START/ {flag=1; next}
    /END/   {flag=0; print ""; next}
    flag    {print}
' >> "$OUTFILE"

echo ""
echo "---------------------------------------------"
echo "Data saved to:   $OUTFILE"
echo "Full log saved:  $LOGFILE"
echo "---------------------------------------------"
