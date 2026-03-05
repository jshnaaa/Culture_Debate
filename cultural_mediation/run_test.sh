#!/bin/bash

################################################################################
# Cultural Conflict Mediation System - Test Script
#
# Usage:
#   ./run_test.sh [MODEL] [SAMPLE_IDX]
#
# Arguments:
#   MODEL       : llama (default) or qwen
#   SAMPLE_IDX  : Sample index to test (default: 0)
#
# Examples:
#   ./run_test.sh                    # Use llama, test sample 0
#   ./run_test.sh llama              # Use llama, test sample 0
#   ./run_test.sh qwen               # Use qwen, test sample 0
#   ./run_test.sh llama 5            # Use llama, test sample 5
#   ./run_test.sh qwen 10            # Use qwen, test sample 10
################################################################################

# Default values
MODEL=${1:-llama}
SAMPLE_IDX=${2:-0}

# Validate MODEL parameter
if [ "$MODEL" != "llama" ] && [ "$MODEL" != "qwen" ]; then
    echo "Error: MODEL must be 'llama' or 'qwen'"
    echo "Usage: $0 [llama|qwen] [sample_idx]"
    exit 1
fi

# Validate SAMPLE_IDX is a number
if ! [[ "$SAMPLE_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: SAMPLE_IDX must be a non-negative integer"
    echo "Usage: $0 [llama|qwen] [sample_idx]"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "Cultural Conflict Mediation System - Test Runner"
echo "================================================================================"
echo "Configuration:"
echo "  MODEL       : $MODEL"
echo "  SAMPLE_IDX  : $SAMPLE_IDX"
echo "================================================================================"
echo ""

# Run the test
python quick_start.py --model "$MODEL" --sample_idx "$SAMPLE_IDX"

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
