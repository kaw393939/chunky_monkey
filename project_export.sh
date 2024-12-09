#!/bin/bash

# Output file for the project export
OUTPUT_FILE="project_export.txt"

# Write project structure to the output file
echo "Project Structure:" > $OUTPUT_FILE
tree -I "__pycache__|*.pyc|*.pyo|.git|venv|*.egg-info" >> $OUTPUT_FILE

# Add a separator
echo -e "\n\nFile Contents:\n" >> $OUTPUT_FILE

# Find and process all files, excluding Python/Git noise
find . -type f \
    ! -path "./__pycache__/*" \
    ! -path "./*.pyc" \
    ! -path "./*.pyo" \
    ! -path "./.git/*" \
    ! -path "./venv/*" \
    ! -path "./*.egg-info" \
    ! -name "*.log" \
    -print | while read file; do
    echo -e "\n\n===== $file =====" >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
done

echo "Project exported to $OUTPUT_FILE"
