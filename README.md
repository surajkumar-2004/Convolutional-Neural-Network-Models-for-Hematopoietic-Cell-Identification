# Blood Cell Classification Using CNN-RNN

A deep learning project for automated white blood cell image classification using a hybrid **Convolutional Neural Network (CNN) + Long Short-Term Memory (LSTM)** architecture.

The project performs image preprocessing, data exploration, augmentation, model training, evaluation, learning-curve visualization, and confusion-matrix analysis using Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, and scikit-learn.

> **Project status:** Academic / experimental machine-learning project. The results in this repository should not be interpreted as a clinical diagnostic system.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Installation](#installation)
- [Project Setup](#project-setup)
- [Running the Notebook](#running-the-notebook)
- [Expected Workflow](#expected-workflow)
- [Outputs](#outputs)
- [Results](#results)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

Blood cell classification is an image-classification problem in which microscopic blood-cell images are assigned to their corresponding cell categories.

This project experiments with a hybrid deep-learning approach:

1. Load and inspect blood-cell images.
2. Read class labels and annotation information.
3. Resize images to a common input size.
4. Normalize pixel values.
5. Split the data into training and test sets.
6. Apply image augmentation.
7. Extract spatial features using a CNN.
8. Extract sequential/grayscale features using LSTM layers.
9. Combine CNN and LSTM representations.
10. Train a softmax classifier.
11. Evaluate predictions using accuracy, classification reports, and confusion matrices.
12. Visualize training behavior and model performance.

---

## Objectives

- Build an image-classification pipeline for blood-cell images.
- Explore preprocessing and data augmentation techniques.
- Compare multi-class and binary classification setups.
- Combine CNN-based spatial feature extraction with LSTM-based feature processing.
- Evaluate the model using standard classification metrics and visualizations.
- Keep the project reproducible and organized for further experimentation.

---

## Model Architecture

The main experimental model uses two feature-extraction branches.

### CNN branch

The CNN branch processes the RGB image using:

- `Conv2D(32, 3x3, ReLU)`
- `Conv2D(64, 3x3, ReLU)`
- `MaxPooling2D`
- `Dropout(0.25)`
- `Flatten`

### LSTM branch

The image is converted to grayscale and passed through:

- `LSTM(64, return_sequences=True)`
- `LSTM(64)`

Dropout and recurrent dropout are used in the LSTM layers.

### Feature fusion

The outputs of the CNN and LSTM branches are concatenated:

```text
Input Image (60 × 80 × 3)
        │
        ├────────────── CNN Branch ──────────────┐
        │                                        │
        │   Conv2D → Conv2D → Pool → Dropout     │
        │                     → Flatten           │
        │                                        │
        └──────────── LSTM Branch ───────────────┤
                 RGB → Grayscale                 │
                 → LSTM(64) → LSTM(64)            │
                                                 │
                         Concatenate              │
                              │                  │
                         Dense(128)              │
                              │                  │
                         Dropout(0.5)             │
                              │                  │
                     Softmax Classification      │
```

The notebook also uses image augmentation such as rotation, zoom, shifts, and horizontal flipping.

---

## Dataset

The uploaded project contains two dataset structures:

### 1. Annotated dataset

Used by the notebook for annotation inspection and image/label processing.

It contains:

- JPEG cell images
- XML annotation files
- CSV labels

### 2. Folder-based image dataset

The project also contains train/test image folders organized by blood-cell subtype.

The uploaded archive contains approximately **12,500 images** in this folder-based dataset and approximately **120 MB** of uncompressed project data overall.

### GitHub recommendation

Do **not** commit the complete raw dataset to the Git repository unless you have confirmed that redistribution is permitted and the repository size is appropriate.

A cleaner repository should contain the code and notebooks while documenting how to obtain the dataset separately.

Add a `DATASET.md` file containing:

- Dataset name
- Original source URL
- License / usage terms
- Download instructions
- Expected local folder structure

Example:

```text
data/
├── dataset-master/
│   └── dataset-master/
│       ├── Annotations/
│       ├── JPEGImages/
│       └── labels.csv/
│
└── dataset2-master/
    └── dataset2-master/
        └── images/
            ├── TRAIN/
            ├── TEST/
            └── TEST_SIMPLE/
```

Update the exact paths in the notebook if your local dataset location differs.

---

## Repository Structure

A professional GitHub version of this project should look like this:

```text
blood-cell-classification/
│
├── README.md
├── .gitignore
├── requirements.txt
├── DATASET.md
├── LICENSE
│
├── notebooks/
│   ├── blood-cell-classification.ipynb
│   └── exploratory-analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   └── reports/
│
└── data/
    └── README.md
```

### Mapping from the uploaded project

The current ZIP is more experimental/notebook-oriented:

```text
Blood cell classification using cnn/
├── scripts/
│   ├── Blood Cells.ipynb
│   └── Untitled.ipynb
│
├── results/
│   ├── Blood.py
│   ├── bloodtk.py
│   └── generated figures
│
└── archive (7)/
    ├── dataset-master/
    └── dataset2-master/
```

For GitHub, I recommend removing the accidental `.ipynb_checkpoints/` directories and avoiding the duplicated/raw dataset folders in the main repository.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep-learning model development |
| OpenCV | Image loading and preprocessing |
| NumPy | Numerical operations |
| Pandas | Data and label processing |
| scikit-learn | Splitting, encoding, metrics, confusion matrix |
| Matplotlib | Visualization |
| Jupyter Notebook | Experimentation and documentation |
| tqdm | Progress monitoring |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/blood-cell-classification.git
cd blood-cell-classification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Jupyter

If it is not already installed:

```bash
pip install notebook
```

Start Jupyter:

```bash
jupyter notebook
```

---

## Project Setup

After downloading the dataset, place it outside Git-tracked files or under an ignored `data/` directory.

For example:

```text
blood-cell-classification/
├── data/
│   └── dataset-master/
│       └── dataset-master/
│           ├── Annotations/
│           ├── JPEGImages/
│           └── labels.csv
│
├── notebooks/
├── results/
└── README.md
```

Then update the notebook paths so they point to your local dataset.

For example:

```python
tree_path = "../data/dataset-master/dataset-master/Annotations"
image_path = "../data/dataset-master/dataset-master/JPEGImages"
```

---

## Running the Notebook

Open the main notebook:

```text
notebooks/blood-cell-classification.ipynb
```

Run the cells in order.

The notebook covers:

1. Library imports
2. Dataset inspection
3. Annotation visualization
4. Label loading
5. Class encoding
6. Image loading
7. Image resizing
8. Pixel normalization
9. Train/test preparation
10. CNN-LSTM model construction
11. Data augmentation
12. Model training
13. Evaluation
14. Learning curves
15. Confusion matrix

---

## Expected Workflow

```text
Raw Dataset
     │
     ▼
Data Inspection
     │
     ▼
Image + Label Extraction
     │
     ▼
Resize to 60 × 80
     │
     ▼
Pixel Normalization
     │
     ▼
Train / Test Split
     │
     ▼
Image Augmentation
     │
     ▼
CNN + LSTM Model
     │
     ▼
Training
     │
     ▼
Prediction
     │
     ├── Accuracy
     ├── Classification Report
     ├── Confusion Matrix
     └── Learning Curves
```

---

## Outputs

The project currently contains several generated figures, including:

- Training accuracy curves
- Training loss curves
- Confusion matrices
- Dataset visualizations
- Blood-cell image examples
- Other exploratory plots

For a cleaner repository, keep final/relevant figures under:

```text
results/
└── figures/
```

Avoid committing temporary notebook output or checkpoint files unless they are intentionally part of the project documentation.

---

## Results

The original project contains experimental plots and evaluation code, but this repository intentionally does not state a single headline accuracy unless the experiment is rerun with a fixed environment, dataset version, preprocessing pipeline, train/test split, and random seed.

When publishing final results, document:

- Dataset version
- Number of training samples
- Number of test samples
- Image size
- Number of classes
- Train/test split
- Number of epochs
- Batch size
- Optimizer
- Learning rate
- Random seed
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Example results table:

| Metric | Value |
|---|---:|
| Test Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| F1-Score | XX.XX% |

Replace the placeholders only after reproducing the experiment.

---

## Important Notes

### 1. This is not a medical diagnostic tool

The project is intended for educational and research purposes. Predictions from an experimental model should not be used as a substitute for professional medical diagnosis.

### 2. Dataset licensing

Before publishing the dataset or redistributing images, verify the original dataset's license and terms of use.

### 3. Reproducibility

For reproducible experiments, record:

- Python version
- TensorFlow/Keras version
- Dataset version
- Random seeds
- Hardware
- Training configuration

### 4. Notebook cleanup

The original project contains Jupyter checkpoint files such as:

```text
.ipynb_checkpoints/
```

These are generated automatically and should normally not be committed.

---

## Future Improvements

Potential improvements include:

- Convert notebook code into reusable Python modules.
- Add a `requirements.txt` with pinned package versions.
- Add deterministic random seeds.
- Add automated train/validation/test splitting.
- Add class-balancing techniques for imbalanced data.
- Compare CNN-only and CNN-LSTM models.
- Add transfer learning using a pretrained architecture such as ResNet or EfficientNet.
- Add precision, recall, F1-score, and ROC-AUC where appropriate.
- Add automated tests for preprocessing and data loading.
- Add experiment tracking.
- Add a simple inference script for classifying a single image.
- Add a web interface using Streamlit or FastAPI.

---

## Contributing

Contributions are welcome.

Suggested workflow:

```bash
git checkout -b feature/your-feature
```

Make your changes, then:

```bash
git add .
git commit -m "Add your feature"
git push -u origin feature/your-feature
```

Open a pull request on GitHub with a clear description of the change.

---



## Author

**Your Name**

- GitHub: `https://github.com/surajkumar-2004`
- LinkedIn: `http://www.linkedin.com/in/suraj-kumar-bala-6064032b6`

---

## Acknowledgements

This project uses publicly available blood-cell image/annotation data. Please credit the original dataset creators and follow the dataset's license and citation requirements.

