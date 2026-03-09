#!/bin/bash

################################################################################
# Cultural Conflict Mediation System - Test Script
#
# Usage:
#   ./run_test.sh [MODEL] [SAMPLE_IDX] [--no_batch]
#
# Arguments:
#   MODEL       : llama (default) or qwen
#   SAMPLE_IDX  : Sample index to test (default: 0)
#   --no_batch  : Generate one agent at a time (saves ~3-4 GB VRAM)
#
# Examples:
#   ./run_test.sh                        # llama, sample 0, batch mode
#   ./run_test.sh llama                  # llama, sample 0, batch mode
#   ./run_test.sh qwen                   # qwen,  sample 0, batch mode
#   ./run_test.sh llama 5                # llama, sample 5, batch mode
#   ./run_test.sh qwen 10                # qwen,  sample 10, batch mode
#   ./run_test.sh llama 0 --no_batch     # llama, sample 0, sequential mode (~17 GB)
#
# VRAM requirements:
#   Batch mode      : ~19-21 GB  (5 prompts generated in parallel)
#   --no_batch mode : ~17 GB     (1 prompt at a time)
################################################################################

# Fix OMP_NUM_THREADS warning from libgomp
export OMP_NUM_THREADS=4

# Default values
MODEL=${1:-llama}
SAMPLE_IDX=${2:-0}
NO_BATCH=${3:-""}

# Validate MODEL parameter
if [ "$MODEL" != "llama" ] && [ "$MODEL" != "qwen" ]; then
    echo "Error: MODEL must be 'llama' or 'qwen'"
    echo "Usage: $0 [llama|qwen] [sample_idx] [--no_batch]"
    exit 1
fi

# Validate SAMPLE_IDX is a number
if ! [[ "$SAMPLE_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: SAMPLE_IDX must be a non-negative integer"
    echo "Usage: $0 [llama|qwen] [sample_idx] [--no_batch]"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "Cultural Conflict Mediation System - Test Runner"
echo "================================================================================"
echo "Configuration:"
echo "  MODEL       : $MODEL"
echo "  SAMPLE_IDX  : $SAMPLE_IDX"
echo "  MODE        : ${NO_BATCH:+sequential (low VRAM)}${NO_BATCH:-batch}"
echo "================================================================================"
echo ""

# Run the test
python quick_start.py --model "$MODEL" --sample_idx "$SAMPLE_IDX" $NO_BATCH

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully (prediction correct)"
else
    echo "✗ Test completed (prediction incorrect or error occurred)"
fi
echo "================================================================================"

exit $EXIT_CODE
