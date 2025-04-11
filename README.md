# Lymphovascular Invasion (LVI) Detection in Breast Cancer Using Deep Learning

This repository contains code and models for detecting Lymphovascular Invasion (LVI) in breast cancer tissues using deep learning techniques. The approach is based on Multiple Instance Learning (MIL) and leverages histological data for the automated classification of cancerous tissues.

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview

Lymphovascular invasion (LVI) is a crucial feature in breast cancer, linked to a higher risk of metastasis and poorer prognosis. Manual detection is labor-intensive and prone to variability. This project uses deep learning models, specifically Swin-Transformer and GigaPath, to automate LVI detection in whole-slide images of breast cancer tissue. Trained on 90 annotated H&E-stained breast cancer slides, the models achieved an AUC of 97%, a sensitivity of 79%, and an average of 8 false positives per slide, demonstrating the potential of these models to improve diagnostic accuracy and consistency.

## Framework

<p align="center">
  <img src="framework.png" alt="Framework">
  <br>
  <em>Figure 1: Overview of the Proposed Framework.</em>
</p>
