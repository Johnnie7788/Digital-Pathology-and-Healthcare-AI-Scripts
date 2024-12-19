
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# Load Dataset (Replace 'data.csv' with actual dataset path)
data = pd.read_csv('data.csv')

# Preprocessing
if 'Readmitted' not in data.columns:
    raise ValueError("Dataset must contain a 'Readmitted' column for prediction.")

# Separating features and target
X = data.drop(columns=['Readmitted'])
y = data['Readmitted']  # Binary column: 1 if readmitted, 0 otherwise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training with Gradient Boosting
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predictions
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

# Evaluation
roc_auc = roc_auc_score(y_test, probabilities)
conf_matrix = confusion_matrix(y_test, predictions)
print(f"ROC AUC: {roc_auc:.2f}")
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", conf_matrix)

# Save evaluation metrics to a file
with open("readmission_metrics.txt", "w") as f:
    f.write(f"ROC AUC: {roc_auc:.2f}\n")
    f.write("Classification Report:\n" + classification_report(y_test, predictions))
    f.write("Confusion Matrix:\n" + str(conf_matrix))

# Automatic Detector and Recommendation System
def automatic_detection(patient_data):
    prediction = best_model.predict(patient_data)
    probability = best_model.predict_proba(patient_data)[:, 1]
    if prediction[0] == 1:
        recommendation = "High likelihood of readmission. Recommend closer monitoring and personalized intervention."
    else:
        recommendation = "Low likelihood of readmission. Maintain standard follow-up protocols."
    return prediction[0], probability[0], recommendation

# Example Detection
sample_patient = X_test.iloc[[0]]  # Example: First patient in test set
prediction, probability, recommendation = automatic_detection(sample_patient)
print(f"Prediction: {'Readmitted' if prediction == 1 else 'Not Readmitted'}")
print(f"Probability: {probability:.2f}")
print(f"Recommendation: {recommendation}")

# SHAP Explainability
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X)

# Save SHAP Summary Plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary_readmission.png")
plt.close()

# Interactive Feature Importance with Plotly
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
fig.write_html("feature_importance.html")

# Dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Patient Readmission Prediction Dashboard"),
    dcc.Graph(id="feature-importance", figure=fig),
    html.Div([
        html.Label("Select SHAP Feature for Analysis:"),
        dcc.Dropdown(
            id="shap-feature-dropdown",
            options=[{"label": col, "value": col} for col in X.columns],
            value=X.columns[0]
        )
    ]),
    dcc.Graph(id="shap-plot"),
    html.Div([
        html.Label("Enter Patient Index for Automatic Detection:"),
        dcc.Input(id="patient-index", type="number", value=0, min=0, max=len(X_test)-1),
        html.Button("Analyze", id="analyze-button", n_clicks=0),
        html.Div(id="detection-output")
    ])
])

@app.callback(
    Output("shap-plot", "figure"),
    [Input("shap-feature-dropdown", "value")]
)
def update_shap_plot(feature):
    shap_values_feature = shap_values[:, X.columns.get_loc(feature)]
    shap_fig = shap.dependence_plot(
        feature, shap_values.values, X, show=False
    )
    plt.savefig("shap_dependence_plot.png")  # Save the plot for reference
    return px.scatter(
        x=X[feature],
        y=shap_values_feature,
        title=f"SHAP Dependence Plot for {feature}",
        labels={"x": feature, "y": "SHAP Value"},
        template="plotly_white"
    )

@app.callback(
    Output("detection-output", "children"),
    [Input("analyze-button", "n_clicks")],
    [Input("patient-index", "value")]
)
def update_detection_output(n_clicks, patient_index):
    if n_clicks > 0:
        patient_data = X_test.iloc[[patient_index]]
        prediction, probability, recommendation = automatic_detection(patient_data)
        result = f"Prediction: {'Readmitted' if prediction == 1 else 'Not Readmitted'}\n"                  f"Probability: {probability:.2f}\n"                  f"Recommendation: {recommendation}"
        return result
    return "Enter a patient index and click Analyze to get results."

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except Exception as e:
        print(f"Error starting the server: {e}")
