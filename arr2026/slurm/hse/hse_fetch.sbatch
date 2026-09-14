#!/bin/bash
#SBATCH --job-name=arr-hsefetch
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/hsefetch_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/hsefetch_%j.err
set -uo pipefail
export TORCHDYNAMO_DISABLE=1
export HF_HOME=/home/lsavchenko/personality-arr/hf
cd /home/lsavchenko/personality-arr
./arrenv/bin/python hse_fetch_repl.py
