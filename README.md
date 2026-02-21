---
language:
- en
license:
- apache-2.0
pretty_name: CSCI218-Bird-Species-Recognition
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
---

# CSCI218 Group Project: Bird Species Recognition
**University of Wollongong | Session 1, 2026**

**Group Details:**
* **Tutorial Group:** T04
* **Group Number:** FT33
* **Project Title:** Bird Species Recognition using CUB-200-2011

**Team Members:**
* Khalis Bin Azman - [9784160]
* Yar Za Bwar - [9959440]
* Jingen Chen - [9086523]
* Zi Yi Ang - [9879900]
* Ryan Shi Jie Ong - [9890956]
* Tarun Kiransskee Boopandar - [9890488]

---

## Project Description
This project attempts to solve a fine-grained image classification problem using the **CUB-200-2011** dataset. The goal is to accurately classify images into one of 200 bird species. This repository contains the code, experiments, and analysis for our group project, specifically comparing and combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

## Dataset Description

- **Homepage:** [CUB 200 2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Repository:** [Caltech Vision Lab](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Paper:** [The Caltech-UCSD Birds-200-2011 Dataset](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf)

### Data Split
|  | Train | Test |
| --- | --- | --- |
| # of observations | 5994 | 5794 |

### Data Instances

A typical data point loaded from the dataset comprises an image and its corresponding species label. 

```python
{
    "image": "<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333>",
    "label": 14,
    "species_name": "015.Lazuli_Bunting"
}

<details>
<summary><strong>Click to expand species mapping</strong></summary>
{
  "0": "001.Black_footed_Albatross",
  "1": "002.Laysan_Albatross",
  "2": "003.Sooty_Albatross",
  "3": "004.Groove_billed_Ani",
  "4": "005.Crested_Auklet",
  "5": "006.Least_Auklet",
  "6": "007.Parakeet_Auklet",
  "7": "008.Rhinoceros_Auklet",
  "8": "009.Brewer_Blackbird",
  "9": "010.Red_winged_Blackbird",
  "10": "011.Rusty_Blackbird",
  "11": "012.Yellow_headed_Blackbird",
  "12": "013.Bobolink",
  "13": "014.Indigo_Bunting",
  "14": "015.Lazuli_Bunting",
  "15": "016.Painted_Bunting",
  "...": "...",
  "195": "196.House_Wren",
  "196": "197.Marsh_Wren",
  "197": "198.Rock_Wren",
  "198": "199.Winter_Wren",
  "199": "200.Common_Yellowthroat"
}

Problem Definition
The problem consists of training a deep learning model to classify instances of the CUB dataset with high accuracy, overcoming the challenges of Fine-Grained Visual Classification (FGVC) such as high inter-class variance and low intra-class variance.

Key research questions include:

How does the local hierarchical feature extraction of a standard CNN (ResNet50) compare to the global context self-attention mechanisms of a Vision Transformer (ViT-B/16)?

Can the severe overfitting risk of a small dataset (~30 images/class) be mitigated effectively using Transfer Learning?

Can performance be improved by combining these contrasting architectures into a Hybrid Ensemble model?

Experimentation Strategy
Given the dataset size constraints and the complexity of fine-grained features, the following strategies were employed:

Transfer Learning: Both models were initialized with pre-trained ImageNet weights to leverage robust, pre-learned feature extractors prior to processing the bird images.

Two-Phase Fine-Tuning: Training the new classification head first with a frozen backbone, followed by unfreezing all layers with a reduced learning rate to adapt features to bird-specific patterns safely.

Multi-Scale Test-Time Augmentation (TTA): Evaluating test images at three different scales (Standard, Wide, and Zoomed) and with horizontal flips to capture microscopic structural details.

Soft-Voting Ensemble: Averaging the softmax probability outputs of both the ResNet50 and ViT-B/16 models to form a final prediction that utilizes the strengths of both architectures.

Evaluation Metric
The primary metric for reporting performance is Top-1 Accuracy on the test set.

Final Project Results:
| Model / Strategy | Final Test Accuracy |
| :--- | :---: |
| ResNet50 (Phase 1: Head Only) | 77.93% |
| ResNet50 (Phase 2: Full Fine-Tuning) | 79.00% |
| ResNet50 (Multi-Scale TTA) | 80.20% |
| Hybrid Ensemble (ResNet50 + ViT-B/16) | 81.86% |

(Detailed classification reports for individual species and confusion matrices can be found in our technical report and Jupyter Notebook).

Dependencies
To run the notebook locally, ensure you have the following libraries installed:

torch (PyTorch)

torchvision

pandas

Pillow (PIL)

matplotlib / seaborn (for visualization)

How to Run
The entire pipeline is contained within a single Jupyter Notebook for ease of execution.

Option 1: Google Colab (Recommended)

Upload the CSCI218_T04_FT33_Bird_Species_Recognition.ipynb file to Google Colab.

Go to Runtime > Change runtime type and select T4 GPU.

Click Runtime > Run all. The notebook will automatically download the dataset, set up the directory structure, and begin training.

Option 2: Local Environment

Clone this repository: git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

Open the .ipynb file in Jupyter Notebook, JupyterLab, or VS Code.

Ensure you have a CUDA-capable GPU available for reasonable training times.

Run the cells sequentially.

Visualizing the Pipeline
Citation Information
Website: CUB200 Dataset

@techreport{WahCUB_200_2011,
	Title = {The Caltech-UCSD Birds-200-2011 Dataset},
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011},
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
