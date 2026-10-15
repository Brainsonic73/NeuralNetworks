# CSC2042S Assignment 4 Report: Neural Networks for Fashion-MNIST Classification

**Student:** Senzo Ntuli  
**Student Number:** ntlsen003  
**Date:** October 2026

---

## 1. Introduction

This assignment investigates a Multi-Layer Perceptron (MLP) for classifying Fashion-MNIST clothing images. The goal is to establish a baseline model, then systematically explore how three hyperparameters — learning rate, batch size, and dropout — affect validation performance. The best configuration is then compared to the baseline to analyse improvements, convergence behaviour, and remaining errors.

---

## 2. Methodology

### 2.1 Data processing

Fashion-MNIST was loaded via `torchvision.datasets.FashionMNIST`. Images were converted to tensors and normalized using the dataset's global mean (0.2860) and standard deviation (0.3530). The 60,000-image training set was split into **50,000 for training and 10,000 for validation**, using a fixed random seed (42). The official 10,000-image test set was reserved for final evaluation. All experiments ran on the UCT `nightmare` server (CPU) in the `perceptron` conda environment (Python 3.12, PyTorch 2.14.0).

### 2.2 Model architecture

A simple MLP with the following structure:
- Input: flattened 28×28 image (784 features)
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU
- Output: 10 neurons (one per class)

Dropout was added between hidden layers when the hyperparameter tuning required it (rate 0.0, 0.2, or 0.5); otherwise the identity function was used.

### 2.3 Training procedure

Models were trained with stochastic gradient descent and cross-entropy loss for **10 epochs**. Training loss and validation loss/accuracy were recorded after each epoch. A validation set of 10,000 held-out images was used to select the best model per configuration.

### 2.4 Hyperparameter experiment design

Three hyperparameters were selected — one covering the **learning process** (learning rate) and two covering **regularisation** (batch size interacts with gradient noise; dropout directly regularises):

| Hyperparameter | Values |
|----------------|--------|
| Learning rate | 0.001, 0.01, 0.1 |
| Batch size | 32, 64, 128 |
| Dropout | 0.0, 0.2, 0.5 |

All 3 × 3 × 3 = **27 combinations** were trained and evaluated. Everything else (architecture, optimiser, epochs, normalisation) was held constant.

---

## 3. Results

### 3.1 Baseline

The baseline model (lr=0.01, bs=64, dropout=0.0) achieved **85.49% final validation accuracy**.

### 3.2 Full grid search results

| LR | Batch | Dropout | Final Val Acc | Best Val Acc |
|----|-------|---------|---------------|--------------|
| 0.001 | 32 | 0.0 | 0.8246 | 0.8246 |
| 0.001 | 32 | 0.2 | 0.8156 | 0.8156 |
| 0.001 | 32 | 0.5 | 0.7864 | 0.7864 |
| 0.001 | 64 | 0.0 | 0.7888 | 0.7888 |
| 0.001 | 64 | 0.2 | 0.7729 | 0.7729 |
| 0.001 | 64 | 0.5 | 0.7442 | 0.7442 |
| 0.001 | 128 | 0.0 | 0.7468 | 0.7468 |
| 0.001 | 128 | 0.2 | 0.7348 | 0.7348 |
| 0.001 | 128 | 0.5 | 0.7182 | 0.7190 |
| 0.01 | 32 | 0.0 | 0.8779 | 0.8779 |
| 0.01 | 32 | 0.2 | 0.8779 | 0.8779 |
| 0.01 | 32 | 0.5 | 0.8645 | 0.8645 |
| 0.01 | 64 | 0.0 | 0.8586 | 0.8586 |
| 0.01 | 64 | 0.2 | 0.8638 | 0.8638 |
| 0.01 | 64 | 0.5 | 0.8479 | 0.8479 |
| 0.01 | 128 | 0.0 | 0.8499 | 0.8499 |
| 0.01 | 128 | 0.2 | 0.8437 | 0.8437 |
| 0.01 | 128 | 0.5 | 0.8309 | 0.8309 |
| 0.1 | 32 | 0.0 | 0.8787 | **0.8842** |
| 0.1 | 32 | 0.2 | 0.8780 | 0.8789 |
| 0.1 | 32 | 0.5 | 0.8724 | 0.8724 |
| 0.1 | 64 | 0.0 | 0.8605 | 0.8753 |
| 0.1 | 64 | 0.2 | 0.8744 | 0.8750 |
| 0.1 | 64 | 0.5 | 0.8573 | 0.8650 |
| 0.1 | 128 | 0.0 | 0.8816 | 0.8830 |
| 0.1 | 128 | 0.2 | 0.8810 | 0.8810 |
| 0.1 | 128 | 0.5 | 0.8665 | 0.8698 |

**Best configuration: lr=0.1, batch_size=32, dropout=0.0 → 88.42% best validation accuracy.**

### 3.3 Visualisations

**Figure 1 — Hyperparameter impact heatmap (batch size × learning rate, three dropout panels):**

![Hyperparameter impact](fig_hyperparam_heatmap.png)

**Figure 2 — Best validation accuracy per learning rate:**

![Learning rate impact](fig_lr_impact.png)

**Figure 3 — Loss curves: baseline vs. best model:**

![Loss curves](fig_loss_curves.png)

**Figure 4 — Confusion matrix for the best model:**

![Confusion matrix](fig_confusion_matrix.png)

---

## 4. Interpretation

### 4.1 Best vs. baseline

The best model (lr=0.1, batch_size=32, dropout=0.0) improved validation accuracy from **85.49% to 88.42%** — an absolute gain of **+2.93 percentage points**. This is a meaningful improvement obtained purely from hyperparameter tuning, without changing the architecture.

### 4.2 Hyperparameter influence

**Learning rate was the most impactful hyperparameter.** At lr=0.001, accuracy ranged from 72–82% — the model **underfit** within the 10-epoch budget because the weight updates were too small to converge. At lr=0.01 and lr=0.1, accuracy jumped to 83–88%. This shows that learning rate directly determines whether the network can reach a good region of the loss surface within the training budget.

**Batch size interacted with learning rate.** At the smallest learning rate, larger batches actually hurt — reaching only ~72–79% — because a small number of large-batch updates gives insufficient gradient signal. At the largest learning rate, larger batches helped slightly, because averaging over more samples reduces the variance of each update and stabilises the large steps.

**Dropout consistently hurt performance.** Across almost every combination, dropout=0.0 outperformed 0.2 and 0.5. This is the opposite of what we might expect if the model were overfitting — but our baseline loss curves show train and validation loss staying close, indicating no significant overfitting. Adding dropout in this regime just slows convergence within the 10-epoch budget, with no regularisation benefit to compensate.

### 4.3 Overfitting analysis

**Figure 3** compares loss curves for the baseline (lr=0.01, bs=64) and the best model (lr=0.1, bs=32). In both cases the training and validation loss remain close throughout the 10 epochs — there is **no severe overfitting**, but there are signs of it beginning:

- The **baseline** curve shows train loss steadily falling from 0.98 to 0.35 while val loss flattens around 0.40. A small gap (≈0.05) opens from epoch 8 onwards.
- The **best model** converges much faster (large learning rate), and its training loss drops further, but its val loss also flattens near 0.35–0.40. The gap is slightly wider than the baseline's at the end.

In short: hyperparameter tuning **did not mitigate overfitting** — it slightly accelerated the onset of the train/val gap because the model converged faster. However, the absolute validation accuracy is still higher, so the trade-off is favourable. More epochs would likely widen the gap; adding dropout *then* might help, but it does not help within 10 epochs.

### 4.4 Error analysis

The **confusion matrix (Figure 4)** shows that the most frequently confused classes are all **upper-body garments**:

| True class | Predicted as | Count |
|-----------|--------------|-------|
| Shirt | T-shirt/top | 176 |
| Shirt | Coat | 108 |
| Shirt | Pullover | 99 |
| T-shirt/top | Shirt | 83 |
| Pullover | Coat | 82 |
| Coat | Pullover | 53 |

Two specific classes, **Shirt and T-shirt/top**, are the most frequently confused with each other. This is visually intuitive: in Fashion-MNIST both are rendered as 28×28 grayscale silhouettes of upper-body garments with similar overall shape. The dataset contains no colour information, no fabric texture, and no sleeve/collar detail at this resolution — the only distinguishing feature is a subtle difference in outline and length. At such low resolution, the differences are often below a pixel of silhouette variation. Similarly, **Coat, Pullover, and Shirt** all share roughly the same torso outline, so any classification error tends to fall within this group rather than across categories (e.g., Shirt → Trouser is essentially never made). This is consistent with the well-known Fashion-MNIST "upper-body garment" confusion pattern documented in the dataset's original paper.

---

## 5. Conclusion

Hyperparameter tuning improved Fashion-MNIST validation accuracy from 85.49% to 88.42%. Learning rate had the strongest effect — too small caused underfitting, too large risked instability. Batch size mattered mainly through its interaction with learning rate. Dropout hurt in this regime because the network was not overfitting within 10 epochs. The remaining errors are concentrated between visually similar upper-body garments (Shirt, T-shirt, Pullover, Coat), which reflects the fundamental ambiguity of the dataset at 28×28 grayscale rather than a failure of the model.

---

## 6. AI usage statement

I used ChatGPT for:
- **Debugging assistance** — resolving PyTorch and nbconvert issues during development
- **Concept explanation** — clarifying the interaction between learning rate, batch size, and gradient noise
- **Code review** — suggestions for structuring the training loop and analysis code
- **Report feedback** — grammar and structure suggestions

All code was written by me. AI assistance was limited to explanation, debugging guidance, and code review as specified by the course's `AGENTS.md` policy. No AI tool generated report paragraphs that were copied verbatim; the analysis and conclusions are my own.