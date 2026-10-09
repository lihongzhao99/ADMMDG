#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

HAC_DATA_ROOT="${HAC_DATA_ROOT:-/path/to/HAC/}"
GPU_ID="${GPU_ID:-0}"

python train_HAC.py --use_flow --use_audio -s human -t animal cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_flow --use_audio -s animal -t human cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_flow --use_audio -s cartoon -t human animal --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_audio -s human -t animal cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_audio -s animal -t human cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_audio -s cartoon -t human animal --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_flow -s human -t animal cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow -s animal -t human cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow -s cartoon -t human animal --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_flow --use_audio -s human -t animal cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow --use_audio -s animal -t human cartoon --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow --use_audio -s cartoon -t human animal --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
