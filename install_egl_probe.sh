#!/bin/bash
# Helper script to install egl-probe with CMake patch
# This is needed for robomimic to work properly

set -e

echo "Installing egl-probe with CMake compatibility patch..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Downloading egl-probe source..."
pip download egl-probe==1.0.2 --no-binary :all:

echo "Extracting..."
tar -xzf egl_probe-1.0.2.tar.gz
cd egl_probe-1.0.2

echo "Patching CMakeLists.txt for modern CMake..."
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' egl_probe/CMakeLists.txt

echo "Building and installing egl-probe..."
pip install .

echo "Cleaning up..."
cd /
rm -rf "$TEMP_DIR"

echo "✓ egl-probe installed successfully!"
