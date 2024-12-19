
# Tumor Classification Using Histopathological Images

This project provides a Python implementation for **Multiclass Tumor Classification** using **Histopathological Images**. The solution leverages a **Convolutional Neural Network (CNN)** based on the **VGG16** architecture and integrates **Grad-CAM** for explainable AI, enabling tumor visualization and detection.

## Features
- **Multiclass Classification**:
  - Classifies tumor types into categories (e.g., benign, malignant, normal).
- **Explainable AI**:
  - Uses Grad-CAM to generate heatmaps that highlight critical regions of interest.
- **Tumor Detection**:
  - Automatically identifies and marks tumor regions in input images.
- **Recommendations**:
  - Provides actionable insights based on classification results.

## File Structure
- `multiclass_tumor_classifier.py`: The main script implementing tumor classification, Grad-CAM visualization, and recommendation generation.

## Requirements
Install the required Python libraries before running the script:
```bash
pip install numpy tensorflow matplotlib opencv-python
```

## Usage
1. Prepare a dataset of histopathological images:
   - Organize images into subdirectories representing each class (e.g., `normal`, `benign`, `malignant`).

2. Update the `data_dir` path in the script to the location of your dataset.

3. Run the script:
   ```bash
   python multiclass_tumor_classifier.py
   ```

4. View the generated heatmaps and marked images to analyze detected tumors.

## Workflow
1. **Data Preprocessing**:
   - Uses `ImageDataGenerator` for data augmentation.
   - Splits the dataset into training and validation subsets.

2. **Model Architecture**:
   - Employs **VGG16** as a feature extractor, followed by fully connected layers for multiclass classification.

3. **Training**:
   - Trains the CNN model using categorical cross-entropy loss and Adam optimizer.

4. **Visualization**:
   - Generates Grad-CAM heatmaps and overlays them on input images for explainability.
   - Detects tumor regions and marks them with bounding boxes.

5. **Recommendation System**:
   - Provides tailored recommendations based on the predicted class and confidence.

## Example Outputs

### Grad-CAM Visualization
Visualizes the regions contributing to the model's predictions.

![Grad-CAM Visualization](example_heatmap.png)

### Tumor Detection with Marker
Identifies and highlights tumor regions in input images.

![Tumor Detection](example_tumor_detection.png)

## Future Enhancements
- Extend support for additional datasets and tumor categories.
- Integrate advanced segmentation techniques for pixel-level tumor detection.
- Deploy as a web application for broader accessibility.

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, reach out at johnjohnsonogbidi@gmail.com.

---

**Tumor Classification Using Histopathological Images** - Empowering AI for medical diagnostics.
