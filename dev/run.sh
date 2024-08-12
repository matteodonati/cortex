#!/bin/bash

# Variables
PROGRAM_SRC="$1"
PROGRAM_NAME=$(basename "$PROGRAM_SRC" .c)
PROGRAM_DIR=$(dirname "$PROGRAM_SRC")
BASE_DIR=$(dirname "$PROGRAM_DIR")
BIN_DIR="$BASE_DIR/bin/$PROGRAM_NAME"

# Create the binary directory if it doesn't exist
mkdir -p "$BIN_DIR" > /dev/null 2>&1

# Uninstall the library silently
sudo make uninstall > /dev/null 2>&1

# Clean project directory silently
make clean > /dev/null 2>&1

# Compile the library silently
make > /dev/null 2>&1

# Install the library silently
sudo make install > /dev/null 2>&1

# Compile the program and place the binary in the specified directory
gcc -o "$BIN_DIR/$PROGRAM_NAME" "$PROGRAM_SRC" -lcortex -L/usr/local/lib -I/usr/local/include/cortex > /dev/null 2>&1

# Run the program
"$BIN_DIR/$PROGRAM_NAME"