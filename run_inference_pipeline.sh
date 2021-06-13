#!/bin/bash

set -euf -o pipefail

python predict.py corrosion
python predict.py edges-and-welds
python merge.py
python.visualize.py

echo "Inference pipeline complete!"
