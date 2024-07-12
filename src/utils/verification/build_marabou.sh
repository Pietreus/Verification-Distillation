#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path_to_Marabou_folder> <path_to_build_folder>"
  exit 1
fi

MARABOU_DIR=$1
BUILD_DIR=$2

# Create the build directory if it does not exist
mkdir -p "$BUILD_DIR"

# Run cmake commands
if ! cmake -S "$MARABOU_DIR" -B "$BUILD_DIR"; then
  echo "CMake configuration failed"
  exit 1
fi

if ! cmake --build "$BUILD_DIR"; then
  echo "CMake build failed"
  exit 1
fi

echo "Build completed successfully."
