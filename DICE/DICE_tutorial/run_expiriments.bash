#!/bin/bash

# Directory to store results
# OUTPUT_DIR="../results/bank/DICE/RQ1"
OUTPUT_DIR="../results/census/DICE/RQ1"
# Base command to run
# COMMAND="python3 DICE_Search.py -dataset=bank -sensitive_index=1 -timeout=600 -RQ=1"
COMMAND="python3 DICE_Search.py -dataset=census -sensitive_index=9 -timeout=600 -RQ=1"

# Type of llm
type="claude"
# type="llama"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"
mkdir -p "$type"

# Loop to run the command 10 times
for i in {1..10}; do
    # Temporary directory for this iteration
    # TEMP_OUTPUT_DIR="${OUTPUT_DIR}/1_10runs"
    TEMP_OUTPUT_DIR="${OUTPUT_DIR}/9_10runs"
    FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${type}/${i}_10runs_${type}"
    mkdir "$TEMP_OUTPUT_DIR"
    
    echo "Running iteration $i..."
    echo "$COMMAND"
    
    # Run the command and store output in a temporary directory
    $COMMAND
    
    # Check if the command completed successfully
    if [ $? -eq 0 ]; then
        # Rename the directory with iteration and type
        if [ -d "$TEMP_OUTPUT_DIR" ]; then
            mv "$TEMP_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"
            echo "Iteration $i completed. Output directory renamed to $FINAL_OUTPUT_DIR."
        else
            echo "Error: Expected output directory $TEMP_OUTPUT_DIR not found."
            echo "Iteration $i failed due to missing output directory." >> "${OUTPUT_DIR}/error_log.txt"
        fi
    else
        echo "Iteration $i failed. Output not saved."
        mv "$TEMP_OUTPUT_DIR" "${FINAL_OUTPUT_DIR}_FAILED"
        echo "Iteration $i failed due to command error." >> "${OUTPUT_DIR}/${type}/error_log.txt"
    fi

done

echo "All iterations completed."
