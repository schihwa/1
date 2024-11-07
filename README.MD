# Image Classification on CIFAR-10 Dataset using MATLAB

## Project Overview

This project involves performing supervised image classification on the CIFAR-10 dataset using MATLAB. The main tasks include:

- Loading and preprocessing the data.
- Selecting target classes.
- Creating training and testing subsets.
- Training multiple machine learning models.
- Performing inference on the testing data.
- Evaluating and comparing results to determine the most suitable model.

## Dataset

- **CIFAR-10 Dataset**: Consists of 60,000 color images (32 × 32 × 3 pixels) in 10 classes, with 6,000 images per class.
- **Provided File**: `cifar-10-data.mat`.
- **Variables**:
  - `data`: 60,000 × 32 × 32 × 3 array holding the images.
  - `labels`: 60,000 × 1 vector holding the corresponding labels.
  - `label_names`: 10 × 1 cell array with the names of each class.

**Viewing an Image Example**:

```matlab
imagesc(squeeze(data(120,:,:,:)));
```

## Project Tasks

### 1. Data Preparation

#### Data Loading

- Load the data into MATLAB.
- Convert images to double data type for processing.

#### Image Visualization

- Create a 1-row, 4-column subplot of randomly chosen images.
- Include the corresponding labels as titles for each subplot.
- Save the figure as a PNG file for reference.

#### Class Selection

- Select **3 classes** pseudo-randomly using `randperm` with a specified seed for reproducibility.
- Store the selected classes in a 3 × 1 vector called `classes`.
- Extract data corresponding to these classes, preserving the original order.
- Resulting data matrix: 18,000 × 32 × 32 × 3.
- Labels vector: 18,000 × 1.

#### Training and Testing Split

- Perform a **50-50** pseudo-randomly selected training-testing split.
- Use `randperm` with the same seed to select 9,000 images for training.
- Store indices in a 9,000 × 1 vector called `training_index`.
- Extract training images and labels; set aside the remaining 9,000 for testing.
- Ensure labels correspond correctly to images.

#### Data Formatting

- Reshape image data into a suitable format for training and testing.
- Create four matrices:
  - **Training data**: 9,000 × 3,072.
  - **Testing data**: 9,000 × 3,072.
  - **Training labels**: 9,000 × 1.
  - **Testing labels**: 9,000 × 1.

#### Label Distribution

- Check the representation of each class in training and testing sets.
- Ensure a balanced distribution for fair evaluation.

### 2. Model Training and Evaluation

#### 2.1 K-Nearest Neighbour (KNN) Implementation

- Classify test data using a custom implementation of KNN with two distance metrics:
  - **L₂-distance** (Euclidean distance).
  - **Another non-Lₚ distance metric** (e.g., Bray-Curtis, Cosine Distance).
- Save prediction results and computation times for evaluation and comparison.

#### 2.2 Existing MATLAB Models

- Train and compare two classification algorithms implemented in MATLAB (excluding KNN). Examples include:
  - **Support Vector Machine (SVM) for Multiclass**: `fitcecoc()`.
  - **Decision Tree**: `fitctree()`.
  - **Ensemble Methods**: `fitcensemble()`.
- Use default parameters or optimize them for better performance.

#### 2.3 Evaluation

- Evaluate models based on:
  - **Accuracy**:
    \[
    \text{Accuracy} = \dfrac{\text{Number of Correct Predictions}}{\text{Total Number of Labels}}
    \]
  - **Confusion Matrix**: To visualize model performance across classes.
  - **Computation Time**: Measure time taken to train and test models using `tic` and `toc`.

- Present results to determine the most suitable model.
- Name variables using the format `modelname_measure` (e.g., `knnL2_accuracy`, `SVM_timetaken`).

### 3. Reporting Results

- Compile findings into a concise report, including:
  - **Introduction**: Overview of the problem and its significance.
  - **Data and Preparation**: Description of the dataset and preprocessing steps, including visualization.
  - **Methodology**: Explanation of model training and evaluation methods, and reasons for model choices.
  - **Results**: Presentation of results in tables and confusion matrices, along with error analysis and observations.
  - **Conclusion**: Recommendation of the most suitable model with justification.

- Ensure the report is clear, well-structured, and free of personal identifiers.

## Getting Started

### Prerequisites

- MATLAB with the Statistics and Machine Learning Toolbox.
- Basic understanding of machine learning concepts and MATLAB programming.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/cifar10-image-classification.git
   ```

2. **Load the Dataset**:

   - Place the `cifar-10-data.mat` file in the project directory.

### Usage

- **Run the Main Script**:

  ```matlab
  main.m
  ```

- The script will execute the following steps:
  - Data loading and preprocessing.
  - Visualization of sample images.
  - Class selection and data splitting.
  - Model training and evaluation.
  - Displaying results and saving output variables.

### Project Structure

- `main.m`: The main script executing the project tasks.
- `data/`: Directory containing the dataset.
- `results/`: Directory where results and figures are saved.
- `report/`: Contains the project report and visualizations.

## Results

- **Accuracy Scores**: Comparison of different models based on accuracy.
- **Confusion Matrices**: Visual representation of model predictions versus actual labels.
- **Computation Times**: Analysis of time efficiency for each model.

## Conclusion

- Summarize the findings and highlight the most effective model for this classification task.
- Discuss potential improvements and future work.

## Contact

For any questions or issues, please open an issue on this repository or contact the maintainer.

---

**Note**: This project is intended for educational purposes and to demonstrate image classification techniques using MATLAB. The CIFAR-10 dataset is a publicly available dataset provided by the Canadian Institute For Advanced Research.