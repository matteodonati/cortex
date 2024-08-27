#!/bin/bash

# Variables
PROGRAM_SRC="$1"
PROGRAM_NAME=$(basename "$PROGRAM_SRC" .c)
PROGRAM_DIR=$(dirname "$PROGRAM_SRC")
BASE_DIR=$(dirname "$PROGRAM_DIR")
BIN_DIR="$BASE_DIR/bin/$PROGRAM_NAME"

# Create the binary directory if it doesn't exist
mkdir -p "$BIN_DIR"

# Uninstall the library 
sudo make uninstall

# Clean project directory 
make clean

# Compile the library 
make

# Install the library 
sudo make install

# Remove existing binary file
rm -f "$BIN_DIR/$PROGRAM_NAME"

# Compile the program and place the binary in the specified directory
gcc -o "$BIN_DIR/$PROGRAM_NAME" "$PROGRAM_SRC" -lcortex -L/usr/local/lib -I/usr/local/include/cortex -lm

# Run the program
if [ $? -eq 0 ]; then
    echo ""
    "$BIN_DIR/$PROGRAM_NAME"
else
    exit 1
fi