# PA1

First programming assignment for FGV EMAp's Deep Learning subject

Instructions: [PDF](https://github.com/Erickslb/deep-learning-fgv-2026/blob/main/PAs/PA1.pdf)

Course materials: [Notion](https://app.notion.com/p/Deep-Learning-2026-2-3bc7a88448a680eeb110ea3cd3931d48)

# Usage

If you have `just` installed, you can also run `just --list` to see the actions below

To download the dataset:
```
curl -L -o ./data/stage1_train.zip https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
unzip -q data/stage1_train.zip -d data/stage1_train
rm ./data/stage1_train.zip
```

To train the model for 20 epochs (GPU strongly recommended):
```bash
uv run scripts/train_resunet_instances.py 20
```

To compute the model's mAP:
```bash
uv run scripts/compute_map.py weights/checkpoint
```
