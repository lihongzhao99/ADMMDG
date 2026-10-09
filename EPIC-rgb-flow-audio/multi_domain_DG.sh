#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPIC_DATA_ROOT="${EPIC_DATA_ROOT:-/path/to/EPIC-KITCHENS/}"
GPU_ID="${GPU_ID:-0}"

python train_EPIC.py --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath "${EPIC_DATA_ROOT}" --seed=0 --gpu="${GPU_ID}" --use_dsu --ema_beta=0.999
