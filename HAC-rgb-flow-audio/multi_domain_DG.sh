#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

HAC_DATA_ROOT="${HAC_DATA_ROOT:-/path/to/HAC/}"
GPU_ID="${GPU_ID:-0}"

python train_HAC.py --use_flow --use_audio -s 'animal' 'cartoon' -t 'human' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_flow --use_audio -s 'human' 'animal' -t 'cartoon' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_flow --use_audio -s 'human' 'cartoon' -t 'animal' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_audio -s 'animal' 'cartoon' -t 'human' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_audio -s 'human' 'animal' -t 'cartoon' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_audio -s 'human' 'cartoon' -t 'animal' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_flow -s 'animal' 'cartoon' -t 'human' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow -s 'human' 'animal' -t 'cartoon' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow -s 'human' 'cartoon' -t 'animal' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_HAC.py --use_video --use_flow --use_audio -s 'animal' 'cartoon' -t 'human' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow --use_audio -s 'human' 'animal' -t 'cartoon' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_HAC.py --use_video --use_flow --use_audio -s 'human' 'cartoon' -t 'animal' --datapath "${HAC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
