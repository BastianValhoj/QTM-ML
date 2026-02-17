#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print commands and their arguments as they are executed (helpful for debugging)
set -x

echo "Starting Siesta build process..."

# -- Pasted tutorial --
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git python3-pip environment-modules
