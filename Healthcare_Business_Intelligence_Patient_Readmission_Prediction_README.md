
# Healthcare Business Intelligence: Patient Readmission Prediction

This project implements a machine learning model for predicting patient readmissions. It integrates advanced techniques like Gradient Boosting, SHAP explainability, and interactive visualizations to provide insights into factors contributing to readmissions and recommendations for medical intervention.

## Features
- **Gradient Boosting Classifier**:
  - Trained and optimized using GridSearchCV for best performance.
- **Automatic Detection and Recommendations**:
  - Predicts if a patient is likely to be readmitted and provides actionable recommendations.
- **SHAP Explainability**:
  - Generates SHAP values to explain feature contributions to predictions.
- **Interactive Dashboard**:
  - Visualizes feature importance, SHAP values, and allows exploration of patient-specific predictions.
- **Evaluation Metrics**:
  - Includes accuracy, ROC AUC, confusion matrix, and classification report.

## Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `plotly`
- `dash`
- `matplotlib`

Install them using:
```bash
pip install pandas numpy scikit-learn shap plotly dash matplotlib
```

## Dataset Requirements
- A CSV file with:
  - **Features**: Clinical and demographic information for prediction.
  - **Readmitted**: Binary column indicating readmission (1 = readmitted, 0 = not readmitted).

Example structure:
| Feature1 | Feature2 | ... | Readmitted |
|----------|----------|-----|------------|
| Value1   | Value2   | ... | 1          |
| Value3   | Value4   | ... | 0          |

Replace `data.csv` with the path to your dataset.

## Usage
1. **Run the Script**:
   ```bash
   python patient_readmission_prediction.py
   ```

2. **Outputs**:
   - **Metrics**: Saved in `readmission_metrics.txt`.
   - **SHAP Summary Plot**: Saved as `shap_summary_readmission.png`.
   - **Feature Importance**: Saved as `feature_importance.html`.

3. **Interactive Dashboard**:
   Access the dashboard at `http://127.0.0.1:8050` in your browser to explore:
   - Feature importance.
   - SHAP values.
   - Patient-specific predictions and recommendations.

## Example
1. **Automatic Detection**:
   Predicts and recommends actions for a sample patient.
   ```python
   Prediction: Readmitted
   Probability: 0.87
   Recommendation: High likelihood of readmission. Recommend closer monitoring and personalized intervention.
   ```

2. **Interactive Visualization**:
   Explore which features contributed most to predictions via SHAP plots and feature importance.

## Contribution
Contributions are welcome! 

## License
This project is licensed under the MIT License.

