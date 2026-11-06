# Human Activity Recognition using Deep Learning

A deep learning project for recognizing human activities from smartphone sensor data using the **UCI Human Activity Recognition (HAR) dataset**.

The project compares traditional machine learning and deep learning approaches and finally uses a **CNN + 561-feature fusion model** to combine raw sensor patterns with the engineered features provided by the dataset.

## Project Overview

The dataset contains sensor recordings collected from smartphones while subjects performed six activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Each activity sample is represented as a fixed-length time window of **128 time steps** with **9 sensor channels**:

- Body acceleration: X, Y, Z
- Body gyroscope: X, Y, Z
- Total acceleration: X, Y, Z

The UCI HAR dataset also provides **561 engineered features** for each window. These features are used as a second source of information in the final fusion model.

## Approach

The project follows a progressive model-development approach:

```text
UCI HAR Dataset
       |
       +----------------------+
       |                      |
Raw Sensor Signals       561 Engineered Features
(128 × 9)                      |
       |                       |
       v                       v
    1D CNN                Dense Feature Branch
       |                       |
       |                       |
       +----------+------------+
                  |
                  v
              Feature Fusion
                  |
                  v
        Dense + Dropout Layers
                  |
                  v
          6-Class Softmax
                  |
                  v
        Activity Prediction
```

### 1. Data Preparation

The official UCI HAR training and test sets are used.

A **subject-independent validation split** is created from the training data using `GroupShuffleSplit`, with subject IDs used as groups. This prevents the same subject from appearing in both training and validation sets.

Final training/validation split:

- Training: **5,551 samples**
- Validation: **1,801 samples**
- Official test set: **2,947 samples**

The raw sensor input has the shape:

```text
(128, 9)
```

The engineered feature input has the shape:

```text
(561,)
```

Both representations are aligned so that the raw sensor window and its corresponding 561 features belong to the same sample.

### 2. Baseline

A **Random Forest** model is used as the traditional machine-learning baseline with the 561 engineered features.

The baseline achieved:

- Accuracy: **92.60%**
- Macro F1: **92.45%**

### 3. 1D CNN

A 1D Convolutional Neural Network is trained directly on the raw sensor sequences.

The CNN learns temporal patterns from the 128 time steps across the 9 sensor channels.

The original CNN achieved:

- Validation accuracy: **97.45%**
- Official test accuracy: **91.65%**
- Official test Macro F1: **91.84%**

### 4. CNN-LSTM

A CNN-LSTM model is also evaluated to combine convolutional feature extraction with recurrent temporal modeling.

It achieved:

- Validation accuracy: **98.00%**
- Official test accuracy: **91.69%**
- Official test Macro F1: **91.62%**

Although its validation performance was high, its official test performance did not improve over the other approaches.

### 5. Final CNN + 561-Feature Fusion

The final model uses two parallel branches.

**CNN branch**

```text
Raw sensor input (128 × 9)
        ↓
Conv1D
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv1D
        ↓
Batch Normalization
        ↓
Conv1D
        ↓
Global Average Pooling
        ↓
Dense
```

**Engineered-feature branch**

```text
561 engineered features
        ↓
Dense
        ↓
Batch Normalization
        ↓
Dropout
```

The outputs of both branches are then concatenated:

```text
CNN representation
       +
561-feature representation
       ↓
Concatenation
       ↓
Dense
       ↓
Dropout
       ↓
Softmax
       ↓
6 activity classes
```

## Final Result

The final fusion model achieved the best official test performance among the models evaluated:

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Random Forest | 92.60% | 92.45% |
| Original 1D CNN | 91.65% | 91.84% |
| CNN-LSTM | 91.69% | 91.62% |
| **CNN + 561 Features** | **94.13%** | **94.15%** |

The final model achieved:

- **Accuracy:** 94.1296%
- **Macro Precision:** 94.6704%
- **Macro Recall:** 94.1600%
- **Macro F1:** 94.1489%

The main remaining classification difficulty is distinguishing **SITTING** from **STANDING**.

## Tech Stack

- Python
- Google Colab
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Dataset

**UCI Human Activity Recognition Using Smartphones Dataset**

The project uses both:

- Raw inertial sensor signals
- The dataset's 561 engineered features

The dataset is used for academic/project purposes and should be cited according to its original publication and dataset license/terms.

## Repository Structure

A typical repository structure can be organized as:

```text
├── data/
├── notebooks/
│   └── HAR_Deep_Learning.ipynb
├── models/
├── results/
├── README.md
└── requirements.txt
```

The exact structure may vary depending on how the project files are organized in the repository.

This project was developed as a team-based deep learning project focused on understanding the complete workflow from data preparation and baseline modeling to deep learning, evaluation, and model fusion.
