#!/bin/bash

dataset="lawschool" # bank, census, lawschool
sensitive_index=4 # bank: 1, census: 8, lawschool: 4

# Directory to store results
OUTPUT_DIR="../results/${dataset}/DICE/RQ1"
# Base command to run
COMMAND="python3 DICE_Search.py -dataset=${dataset} -sensitive_index=${sensitive_index} -timeout=600 -RQ=1"

# Type of llm
type="llama" # llama, claude

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"
mkdir -p "$type"

# Loop to run the command 10 times
for i in {1..10}; do
    # Temporary directory for this iteration
    TEMP_OUTPUT_DIR="${OUTPUT_DIR}/${sensitive_index}_10runs"
    FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${type}/${i}_10runs_${type}"
    mkdir -p "$TEMP_OUTPUT_DIR"
    
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
        mkdir -p "${OUTPUT_DIR}/${type}"
        mv "$TEMP_OUTPUT_DIR" "${FINAL_OUTPUT_DIR}_FAILED"
        echo "Iteration $i failed due to command error." >> "${OUTPUT_DIR}/${type}/error_log.txt"
    fi

done

echo "All iterations completed."
