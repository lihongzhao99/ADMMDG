# ADMMDG

Official PyTorch implementation of **Multimodal Domain Generalization via
Adaptive Dual-Objective Feature Learning**.

## Environment

The code was developed with:

```text
Python 3.10.4
torch 1.11.0+cu113
mmcv-full 1.2.7
mmaction2 0.13.0
```

Additional dependencies are `soundfile`, `scipy`, `imageio`, and `tqdm`.

## Pretrained models

Download the following files into the `pretrained_models` directory of both
`EPIC-rgb-flow-audio` and `HAC-rgb-flow-audio`:

- [SlowFast RGB](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth)
- [SlowOnly Flow](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth)
- [VGGSound Audio](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), renamed to `vggsound_avgpool.pth.tar`

## EPIC-Kitchens

Prepare the dataset as follows:

```text
EPIC-KITCHENS
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
│   ├── train
│   └── test
├── flow
│   ├── train
│   └── test
└── audio
    ├── train
    └── test
```

Set `EPIC_DATA_ROOT` and run all multi-source modality combinations:

```bash
export EPIC_DATA_ROOT=/path/to/EPIC-KITCHENS/
bash EPIC-rgb-flow-audio/multi_domain_DG.sh
```

For single-source domain generalization, train on one source domain and evaluate
the other two target domains:

```bash
bash EPIC-rgb-flow-audio/single_domain_DG.sh
```

Run one experiment directly:

```bash
cd EPIC-rgb-flow-audio
python train_EPIC.py \
  --use_video --use_flow --use_audio \
  -s D2 D3 -t D1 \
  --datapath /path/to/EPIC-KITCHENS/ \
  --use_dsu --gpu 0
```

## HAC

Download the [HAC dataset](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main)
and arrange it as follows:

```text
HAC
├── HAC_Splits
├── human
│   ├── videos
│   ├── flow
│   └── audio
├── animal
│   ├── videos
│   ├── flow
│   └── audio
└── cartoon
    ├── videos
    ├── flow
    └── audio
```

Set `HAC_DATA_ROOT` and run all multi-source modality combinations:

```bash
export HAC_DATA_ROOT=/path/to/HAC/
bash HAC-rgb-flow-audio/multi_domain_DG.sh
```

For single-source domain generalization:

```bash
bash HAC-rgb-flow-audio/single_domain_DG.sh
```

Run one experiment directly:

```bash
cd HAC-rgb-flow-audio
python train_HAC.py \
  --use_video --use_flow --use_audio \
  -s animal cartoon -t human \
  --datapath /path/to/HAC/ \
  --use_dsu --gpu 0
```

Set `GPU_ID` before invoking either shell script to select a GPU; it defaults to
`0`.

PCGrad is enabled by default to resolve gradient conflicts among classification,
MDIL, MSDIL, and feature-separation objectives. Add `--disable_pcgrad` to run an
ablation with standard summed-gradient optimization.

Checkpoints are saved under `models/single_source` or `models/multi_source`.
Training logs are saved under `results/single_source` or
`results/multi_source`.

## Acknowledgement

We thank the [SimMMDG](https://github.com/donghao51/SimMMDG) project for its
valuable open-source implementation.
