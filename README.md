# SoM2026 Baseline (Official)

> This repository provides the official baseline implementation for the **SoM Challenge 2026**. It includes the baseline model, training and evaluation scripts, instructions for reproducing the baseline results, and the expected submission format.

## 1) Challenge Overview

**SoM Challenge 2026: Wireless Foundation Model-Empowered Multi-Modal Sensing and Communications** aims to further investigate the potential of wireless foundation models for multi-modal sensing and communication tasks via Synesthesia of Machines (SoM).

### Tasks

* **Task 1 – LoS/NLoS scenario classification:** predict whether a radio link contains a Line-of-Sight path.
* **Task 2 – Multi-modal-enhanced channel prediction:** predict the CSI over the remaining $$K-K_1$$ subcarriers from the CSI over the first $$K_1$$ subcarriers, surrounding RGB images observed by the UE, and its location.
* **Task 3 – Multi-modal-enhanced depth map estimation:** estimate the depth map of the vehicle’s front perspective from the RGB image observed from the same perspective and CSI between the base station and the vehicle.

### Data Source and Settings

| Item                | Task 1        | Task 2        | Task 3        |
| ------------------- | ------------- | ------------- | ------------- |
| Generator           | QuaDRiGa      | SynthSoM      | SynthSoM      |
| Scenario            | UMi / UMa     | Cross road    | Forking road  |
| Training Samples    | 10            | 500           | 1000          |
| Input Dim.          | (24, 8, 128)  | CSI: (128, 64)<br>RGB: (3, 224, 224)<br>Position: (2) | CSI: (32, 64)<br>RGB: (3, 224, 224) |
| Output Dim.         | (2)           | (128, 64)     | (1, 224, 224) |
| Config.             | SNR = 15 dB   | RSF position: (-72.4, 165.7)<br>SNR = 20 dB | RSF position: (175.6, -49.3) |
| Metric              | F1 score      | NMSE          | MAE           |

> For more details about **SynthSoM** (including scenario description, sensor parameter, RSF position, etc.), please refer to [SynthSoM](https://github.com/PKU-PCNI/SynthSoM).

---

## 2) Dataset Structure

Download the official SoM2026 dataset from [Dataset](https://huggingface.co/datasets/pku-pcni-lab/WiFo-2-SoM-Challenge) and place it under `./dataset/`.

The recommended directory structure is:

```text
SoM2026-baseline/
├── dataset/
│   ├── Task1/
│   │   ├── H_train.mat
│   │   └── L_train.mat
│   ├── Task2/
│   │   ├── X_train_prev.mat
│   │   ├── X_train.mat
│   │   ├── imgs_train.mat
│   │   └── location_train.mat
│   └── Task3/
│       ├── H_train.mat
│       ├── RGB_train.mat
│       └── depth_train.mat
├── weights/
├── DataLoader.py
├── ...
```

---

## 3) Getting Started

### 3.1 Clone the Repository

```bash
git clone https://github.com/JiashuoTang/SoM2026-baseline.git
cd SoM2026-baseline
```

### 3.2 Environment

We recommend using Python 3.10+ and CUDA-enabled PyTorch.

```bash
# (Optional) Create a new environment
conda create -n som2026 python=3.10 -y
conda activate som2026

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 3.3 Download the Dataset

Download the official dataset from [Dataset](https://huggingface.co/datasets/pku-pcni-lab/WiFo-2-SoM-Challenge) and place it under `./dataset/`.

### 3.4 Download the Pre-trained Weights

Download the pre-trained **WiFo-2** weights from [WiFo-2](https://huggingface.co/pku-pcni-lab/WiFo-2) and place it under `./weights/`.

---

## 4) Run the Baseline

### 4.1 Task 1

```bash
python main.py --task_id 1
```

### 4.2 Task 2

```bash
python main.py --task_id 2
```

### 4.3 Task 3

```bash
python main.py --task_id 3
```

---

## 5) Expected Results

Using the official dataset and the default baseline, the test performance is approximately:

| Task              | Metric         | Baseline Result |
| ----------------- | -------------- | --------------- |
| Task 1            | F1 score       | 0.62            |
| Task 2            | NMSE           | 0.128           |
| Task 3            | MAE (m)        | 9.79            |

> Minor deviations may occur because of differences in random seeds, hardware, CUDA versions, and software environments.

---

## 6) Submission Format

### 6.1 Prediction File

The expected format can be generated using the following Python script:

```python
import json
import numpy as np

N1 = N2 = N3 = 4
# Load or compute your predictions here:
pred_t1 = np.random.randint(0, 2, size=(N1,))   # classes or probabilities
pred_t2 = np.random.randn(N2, 2, 128, 64)       # full channel reconstructions
pred_t3 = np.random.randn(N3, 1, 224, 224)      # (1, H, W) in meters
# Model complexity
param = 5.45
flops = 1.54

payload = {
    "task1": pred_t1.tolist(),
    "task2": pred_t2.tolist(),
    "task3": pred_t3.tolist(),
    "param": param,
    "flops": flops,
}
with open("submission.json", "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
```

> Make sure that the order of predictions exactly matches the order of samples in the official test set.

### 6.2 Model Complexity

The number of trainable parameters and FLOPs should be calculated using the **same** implementation and `calflops` as `flops.py`.

Run the following command to obtain the model complexity results for this baseline (see `flops_log.txt`):

```bash
python flops.py
```

---

## 7) License

This project is released under the **Apache 2.0** License (unless otherwise specified in the repo).

---

## 8) Contact

For questions or issues, please open a GitHub Issue or reach the organizers at:

* [boxunliu@stu.pku.edu.cn](boxunliu@stu.pku.edu.cn)
* [xyliu25@stu.pku.edu.cn](xyliu25@stu.pku.edu.cn)
* [tangjiashuo26@stu.pku.edu.cn](tangjiashuo26@stu.pku.edu.cn)
  
