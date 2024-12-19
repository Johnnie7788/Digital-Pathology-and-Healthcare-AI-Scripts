
# Predictive Modeling of Patient Outcomes in Oncology

This project demonstrates a comprehensive approach to predictive modeling for cancer patient outcomes using clinical and genomic datasets. The workflow includes machine learning, survival analysis, explainability, and interactive visualizations.

## Features
- **Random Forest Classifier**: Predicts survival outcomes based on patient data.
- **Kaplan-Meier Survival Curves**: Visualizes survival probabilities for risk groups.
- **SHAP Explainability**: Explains model predictions and feature importance.
- **Interactive Dashboard**: Enables dynamic exploration of survival probabilities and feature distributions.
- **Log-rank Test**: Compares survival distributions between risk groups statistically.

## Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `lifelines`
- `plotly`
- `dash`
- `shap`
- `matplotlib`

Install them using:
```bash
pip install pandas numpy scikit-learn lifelines plotly dash shap matplotlib
```

## Dataset
Prepare a dataset with the following structure:
- **SurvivalTime**: Time (in days) until the event or last follow-up.
- **Event**: Binary outcome (1 = event occurred, 0 = censored).
- **Features**: Clinical and genomic variables used for modeling.

Example file structure:
```
data.csv
```

## Usage
1. **Run the Script**:
   Replace `'data.csv'` with the path to your dataset and execute the Python script:
   ```bash
   python predictive_modeling_oncology.py
   ```

2. **Outputs**:
   - **Evaluation Metrics**: Saved to `evaluation_metrics.txt`.
   - **SHAP Summary Plot**: Saved as `shap_summary_plot.png`.
   - **Kaplan-Meier Curves**: Displayed in the dashboard.
   - **Log-rank Test Results**: Saved to `logrank_test_results.txt`.

3. **Interactive Dashboard**:
   Launch the dashboard to explore survival probabilities and feature distributions:
   ```bash
   python predictive_modeling_oncology.py
   ```

   Access it at `http://127.0.0.1:8050` in your browser.

## Model Overview
- **Machine Learning**: Random Forest with optimized hyperparameters.
- **Survival Analysis**: Kaplan-Meier curves and log-rank tests.
- **Explainability**: SHAP values for feature importance.

## Example Visualizations
1. **Kaplan-Meier Curves**:
   Visualizes survival probabilities for low-risk and high-risk groups.
2. **SHAP Summary Plot**:
   Highlights the most important features contributing to predictions.

## Contribution
Contributions are welcome! 

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the open-source community for tools like `scikit-learn`, `lifelines`, `plotly`, and `shap` that make such projects possible.
