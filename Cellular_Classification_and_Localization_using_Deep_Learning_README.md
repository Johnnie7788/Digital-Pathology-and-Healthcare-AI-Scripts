
# Cellular Classification and Localization using Deep Learning

This project implements a deep learning pipeline for cellular classification and localization from high-resolution biological images. It leverages **EfficientNetB0** as a base model and includes Grad-CAM for explainability, making it suitable for biomedical and healthcare applications.

## Features
1. **Deep Learning Architecture**:
   - Uses **EfficientNetB0** for transfer learning.
   - Fine-tuned layers for cellular classification.
   
2. **Explainability with Grad-CAM**:
   - Visualizes areas contributing to model predictions for interpretability.

3. **Evaluation Metrics**:
   - Provides comprehensive metrics such as Accuracy, F1-Score, Precision, Recall, and Confusion Matrix.

4. **Data Augmentation**:
   - Includes rotation, zooming, shifting, and flipping for robust model training.

5. **Visualization**:
   - Outputs training accuracy plots and Grad-CAM heatmaps.

6. **Deployment Ready**:
   - Saves the trained model in `.h5` format for easy deployment.

## Prerequisites
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `cv2`
- `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas tensorflow matplotlib opencv-python scikit-learn
```

## Dataset Requirements
- A **CSV file** containing:
  - `image_name`: Image filenames.
  - `label`: Corresponding labels for classification.
- High-resolution biological images stored in a directory.

Example structure of the `labels.csv`:
| image_name      | label |
|------------------|-------|
| cell_image_1.png | 0     |
| cell_image_2.png | 1     |

Update the dataset paths in the script:
```python
data_dir = "./biological_images"
labels_file = "./labels.csv"
```

## Usage
1. **Run the Script**:
   ```bash
   python cellular_classification_localization.py
   ```

2. **Outputs**:
   - Trained Model: `cellular_classification_model.h5`
   - Training Accuracy Plot: `training_accuracy.png`
   - Grad-CAM Heatmap: `grad_cam_output.png`

3. **Evaluate the Model**:
   - Metrics such as F1-Score, Precision, and Recall are displayed in the console.

## Example
### Grad-CAM Output:
![Grad-CAM Example](grad_cam_output.png)

### Training Accuracy Plot:
![Training Accuracy](training_accuracy.png)

## Contribution
Contributions are welcome! 

## License
This project is licensed under the MIT License.


