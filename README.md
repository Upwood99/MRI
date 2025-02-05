# README 

## Overview

This Jupyter Notebook, `IRM_Processing.ipynb`, is designed to process and analyze chest X-ray images for the purpose of detecting pneumonia. The notebook performs Exploratory Data Analysis (EDA) on a dataset of chest X-ray images, which includes both normal and pneumonia cases. The dataset is divided into training, testing, and validation sets.

## Project Structure

The notebook is structured as follows:

1. **Importing Libraries**: The necessary libraries such as `cv2`, `pandas`, `numpy`, `matplotlib`, and `tensorflow` are imported to handle image processing, data manipulation, visualization, and deep learning tasks.

2. **Data Path Setup**: The paths to the dataset directories (training, testing, and validation) are defined.

3. **Exploratory Data Analysis (EDA)**:
   - The notebook begins by counting the number of images in each category (normal and pneumonia) for the training, testing, and validation sets.
   - It then visualizes a sample of images from both categories using `matplotlib`.

4. **Data Preprocessing**:
   - The notebook uses `ImageDataGenerator` from `tensorflow.keras.preprocessing.image` to preprocess the images. This includes rescaling the images and preparing them for input into a deep learning model.

5. **Model Training**:
   - Although the actual model training code is not included in the provided snippet, the notebook is set up to use a deep learning model (likely a Convolutional Neural Network) for image classification. The `ImageDataGenerator` is used to create batches of image data for training and validation.

6. **Visualization**:
   - The notebook includes code to visualize the distribution of images across different categories using `seaborn`.

## Dataset

The dataset used in this notebook is organized into three main directories:
- **Training Data**: Contains images for training the model, further divided into `NORMAL` and `PNEUMONIA` subdirectories.
- **Testing Data**: Contains images for testing the model, also divided into `NORMAL` and `PNEUMONIA` subdirectories.
- **Validation Data**: Contains images for validating the model, with the same subdirectory structure.

## Requirements

To run this notebook, you need the following Python libraries installed:
- `cv2` (OpenCV)
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`

You can install these libraries using `pip`:

```bash
pip install opencv-python pandas numpy matplotlib seaborn tensorflow
```

## Usage

1. **Clone the Repository**: Clone the repository containing the notebook and dataset.

2. **Set Up the Environment**: Ensure all required libraries are installed.

3. **Run the Notebook**: Open the notebook in Jupyter and run each cell sequentially to perform the EDA, data preprocessing, and model training.

4. **Model Training**: If the model training code is not included, you can add your own deep learning model and train it using the preprocessed data.

## Notes

- The notebook is designed to be run in a Jupyter environment.
- Ensure that the dataset paths are correctly set up in the notebook to match your local directory structure.
- The notebook is currently set up for binary classification (normal vs. pneumonia). If you wish to extend it for multi-class classification, you will need to modify the data loading and preprocessing steps accordingly.

## Acknowledgments

- The dataset used in this notebook is from the Chest X-Ray Images (Pneumonia) dataset available on Kaggle.
- Special thanks to the open-source community for providing the libraries and tools used in this project.

## Contact

For any questions or suggestions, please contact the project maintainer at zakaria.oukanna@hotmail.com.
