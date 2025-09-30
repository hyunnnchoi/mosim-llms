#!/bin/bash

# Convert all Kineto JSON traces to Chakra ET format

echo "=========================================="
echo "Chakra ET Converter"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "✗ Error: Python not found"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

# Run conversion script
$PYTHON_CMD convert_to_et.py

echo ""
echo "Done!"
