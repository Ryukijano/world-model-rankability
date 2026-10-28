---
title: World-model rankability (live)
emoji: 🦾
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
license: cc-by-nc-sa-4.0
short_description: Live RoboJEPA-22M action ranking on ZeroGPU
startup_duration_timeout: 1h
---

Live companion to the [world-model rankability benchmark](https://huggingface.co/spaces/Ryukijano/world-model-rankability).
Runs Meta's released RoboJEPA-22M (150k) with the frozen V-JEPA 2.1 ViT-G/384 encoder on ZeroGPU to rank 64 candidate
actions per benchmark state, and compares the ranking with simulator ground truth (MuJoCo replay, RoboCasa).

Vendored code under `vendor/`: [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa)
(CC BY-NC-SA 4.0) and [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) (see the LICENSE files there).
Non-commercial use only.
