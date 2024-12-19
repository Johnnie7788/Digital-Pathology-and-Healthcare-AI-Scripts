
# Automated Tumor Segmentation for Digital Pathology

This project implements a U-Net-based deep learning model for automated tumor segmentation using histopathological images. The model is trained on paired image and mask datasets and leverages advanced techniques such as data augmentation, dropout regularization, and batch normalization to achieve robust segmentation performance.

## Features
- **U-Net Architecture**: Encoder-decoder structure with skip connections.
- **Data Augmentation**: Random horizontal and vertical flips for better generalization.
- **Dice Loss**: Custom loss function optimized for segmentation tasks.
- **Visualization**: Displays input image, predicted mask, and overlay for interpretation.

## Installation

### Prerequisites
Ensure you have Python 3.7 or above and the following libraries installed:
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

Install the required Python packages using pip:
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

## Usage

### Dataset Preparation
Prepare a dataset with the following structure:
```
segmentation_data/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...
```

### Training
Run the Python script to train the model:
```bash
python unet_tumor_segmentation.py
```

### Visualization
The script generates visualizations of the input image, predicted mask, and overlay after training.

## Model Architecture
The U-Net model consists of:
- **Encoder**: Convolutional layers with batch normalization and max pooling.
- **Bottleneck**: High-dimensional feature extraction layers.
- **Decoder**: Up-sampling layers with skip connections to recover spatial resolution.

## Callbacks
The training process includes:
- **EarlyStopping**: Stops training when validation loss stops improving.
- **ModelCheckpoint**: Saves the best model during training.
- **CSVLogger**: Logs training progress to a CSV file.
- **TensorBoard**: Logs training metrics for visualization.

## Results
After training, the model saves:
- The best-performing model as `unet_tumor_segmentation_best_model.h5`.
- A visualization of segmentation predictions.

## Contributions
Feel free to contribute to this project

## License
This project is licensed under the MIT License.
