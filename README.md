# LVI-Detection

This repository contains code and models for detecting Lymphovascular Invasion (LVI) in breast cancer tissues using deep learning techniques. 

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)

## Overview

LVI is a crucial feature in breast cancer, linked to a higher risk of metastasis and poorer prognosis. Manual detection is labor-intensive and prone to variability. This project uses deep learning models, specifically Swin-Transformer and GigaPath, to automate LVI detection in whole-slide images of breast cancer tissue. Trained and evaluated on 90 annotated H&E-stained breast cancer slides, the best model achieved an AUC of 97%, a sensitivity of 79%, and an average of 8 false positives per slide, demonstrating the potential of these models to improve diagnostic accuracy and consistency.

## Framework

<p align="left">
  <img src="framework.png" alt="Framework">
  <br>
  <em>Figure 1: Overview of the Proposed Framework. (a) Training Phase: (I) LVI foci are annotated by two pathologists, with a third resolving disagreements. (II) Swin-Small uses extracted patches for fine-tuning. (III) GigaPath tessellates the WSI and extracts patch embeddings for binary classification. (b) Inference Phase: (1) Swin-Small predicts patch-wise probabilities using tessellation and sliding window, with postprocessing to identify LVI. (2) GigaPath computes patch probabilities in one step, followed by postprocessing to locate LVI.</em>
</p>

## Results

<p align="left">
  <img src="swin_8.png" alt="swin_8" width="400" style="margin-right: 20px;">
  <img src="giga_8.png" alt="giga_8" width="400">
  <br>
  <em>Figure 2: Left: A WSI with predictions from Swin-Small. Right: The same WSI with predictions from GigaPath. The predicted LVI locations are highlighted with red boxes, while the ground-truth LVI locations are marked with green boxes.</em>
</p>

## Acknowledgements
This project uses the code and pre-trained weights provided by [GigaPath](https://github.com/prov-gigapath/prov-gigapath) and [timm](https://github.com/huggingface/pytorch-image-models). We have modified the original code to adapt it for our specific task of LVI detection in breast cancer whole-slide images.

We thank the GigaPath and timm library teams for making their code and model weights publicly available.


## Reference

Please consider citing the following paper if you find our work useful for your project.


```
@InProceedings{,
  title = {},
  author = {},
  booktitle = {},
  pages = {},
  year = {2025},
  volume = {},
  series = {},
  month = {},
  publisher = {},
}
```
