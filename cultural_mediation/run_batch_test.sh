#!/bin/bash

################################################################################
# Cultural Conflict Mediation System - Batch Test Script
#
# Tests multiple samples and computes overall accuracy
#
# Usage:
#   ./run_batch_test.sh [MODEL] [NUM_SAMPLES]
#
# Arguments:
#   MODEL        : llama (default) or qwen
#   NUM_SAMPLES  : Number of samples to test (default: 10)
#
# Examples:
#   ./run_batch_test.sh                # Test 10 samples with llama
#   ./run_batch_test.sh llama 20       # Test 20 samples with llama
#   ./run_batch_test.sh qwen 50        # Test 50 samples with qwen
################################################################################

# Default values
MODEL=${1:-llama}
NUM_SAMPLES=${2:-10}

# Validate MODEL parameter
if [ "$MODEL" != "llama" ] && [ "$MODEL" != "qwen" ]; then
    echo "Error: MODEL must be 'llama' or 'qwen'"
    echo "Usage: $0 [llama|qwen] [num_samples]"
    exit 1
fi

# Validate NUM_SAMPLES is a number
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -lt 1 ]; then
    echo "Error: NUM_SAMPLES must be a positive integer"
    echo "Usage: $0 [llama|qwen] [num_samples]"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "Cultural Conflict Mediation System - Batch Test Runner"
echo "================================================================================"
echo "Configuration:"
echo "  MODEL        : $MODEL"
echo "  NUM_SAMPLES  : $NUM_SAMPLES"
echo "================================================================================"
echo ""

# Initialize counters
CORRECT=0
TOTAL=0

# Run tests
for ((i=0; i<NUM_SAMPLES; i++)); do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Testing sample $i / $NUM_SAMPLES"
    echo "--------------------------------------------------------------------------------"

    # Run test
    python quick_start.py --model "$MODEL" --sample_idx "$i" > /dev/null 2>&1

    # Check result
    if [ $? -eq 0 ]; then
        CORRECT=$((CORRECT + 1))
        echo "✓ Sample $i: CORRECT"
    else
        echo "✗ Sample $i: INCORRECT"
    fi

    TOTAL=$((TOTAL + 1))
done

# Compute accuracy
ACCURACY=$(echo "scale=4; $CORRECT / $TOTAL * 100" | bc)

# Print results
echo ""
echo "================================================================================"
echo "Batch Test Results"
echo "================================================================================"
echo "Model         : $MODEL"
echo "Total Samples : $TOTAL"
echo "Correct       : $CORRECT"
echo "Incorrect     : $((TOTAL - CORRECT))"
echo "Accuracy      : $ACCURACY%"
echo "================================================================================"

# Save results to file
RESULT_FILE="results/batch_test_${MODEL}_${NUM_SAMPLES}samples.txt"
mkdir -p results
cat > "$RESULT_FILE" <<EOF
Cultural Conflict Mediation System - Batch Test Results
========================================================
Date: $(date)
Model: $MODEL
Total Samples: $TOTAL
Correct: $CORRECT
Incorrect: $((TOTAL - CORRECT))
Accuracy: $ACCURACY%
========================================================
EOF

echo ""
echo "✓ Results saved to: $RESULT_FILE"
echo ""
