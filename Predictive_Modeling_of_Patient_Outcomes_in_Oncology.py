
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import shap
import matplotlib.pyplot as plt
import os

# Load dataset with error handling
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Dataset file 'data.csv' not found. Please provide a valid dataset.")

# Preprocessing
# Assumes 'SurvivalTime', 'Event', and clinical/genomic feature columns are present
if 'SurvivalTime' not in data.columns or 'Event' not in data.columns:
    raise ValueError("Dataset must contain 'SurvivalTime' and 'Event' columns.")

survival_time_col = 'SurvivalTime'
event_col = 'Event'
feature_cols = [col for col in data.columns if col not in [survival_time_col, event_col]]

# Split data into features and target
X = data[feature_cols]
y = data[event_col]

# Train-test split
if len(data) < 20:
    raise ValueError("Dataset is too small for train-test split. Ensure dataset has sufficient rows.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model building
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
predictions = rf_model.predict(X_test)
pred_probs = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, pred_probs)
conf_matrix = confusion_matrix(y_test, predictions)

# Save evaluation metrics to a file
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"ROC AUC: {roc_auc:.2f}\n")
    f.write("Classification Report:\n" + classification_report(y_test, predictions))
    f.write("Confusion Matrix:\n" + str(conf_matrix))

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", conf_matrix)

# Feature Importance with SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Save SHAP Summary Plot
shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
plt.savefig("shap_summary_plot.png")
plt.close()

# Kaplan-Meier Survival Curve
data['RiskGroup'] = pd.cut(
    rf_model.predict_proba(data[feature_cols])[:, 1], bins=[0, 0.5, 1], labels=['Low Risk', 'High Risk']
)
kmf = KaplanMeierFitter()
fig = go.Figure()

for group in data['RiskGroup'].unique():
    group_data = data[data['RiskGroup'] == group]
    kmf.fit(group_data[survival_time_col], event_observed=group_data[event_col], label=str(group))
    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_[kmf.survival_function_.columns[0]],
        mode='lines',
        name=str(group)
    ))

# Log-rank test
low_risk_data = data[data['RiskGroup'] == 'Low Risk']
high_risk_data = data[data['RiskGroup'] == 'High Risk']
logrank_results = logrank_test(
    low_risk_data[survival_time_col], high_risk_data[survival_time_col],
    event_observed_A=low_risk_data[event_col], event_observed_B=high_risk_data[event_col]
)
with open("logrank_test_results.txt", "w") as f:
    f.write(f"Log-rank test p-value: {logrank_results.p_value:.5f}\n")

print(f"Log-rank test p-value: {logrank_results.p_value:.5f}")

fig.update_layout(
    title="Kaplan-Meier Survival Curves",
    xaxis_title="Time (days)",
    yaxis_title="Survival Probability",
    template="plotly_white"
)

# Dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Predictive Modeling of Patient Outcomes in Oncology"),
    dcc.Graph(id="km-curve", figure=fig),
    html.Div([
        html.Label("Select Feature for Visualization:"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": col, "value": col} for col in feature_cols],
            value=feature_cols[0]
        )
    ]),
    dcc.Graph(id="feature-distribution")
])

@app.callback(
    Output("feature-distribution", "figure"),
    [Input("feature-dropdown", "value")]
)
def update_feature_distribution(feature):
    fig = go.Figure()
    for group in data['RiskGroup'].unique():
        fig.add_trace(go.Box(
            y=data[data['RiskGroup'] == group][feature],
            name=str(group),
            boxmean=True
        ))
    fig.update_layout(
        title=f"Distribution of {feature} by Risk Group",
        yaxis_title=feature,
        template="plotly_white"
    )
    return fig

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except Exception as e:
        print(f"Error starting the server: {e}")
