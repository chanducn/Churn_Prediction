from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and reference DataFrame (for columns)
model = pickle.load(open("churn-flask-app/model/Rf_churn_model.sav", "rb"))
df_ref = pd.read_csv("tele_churn_1.csv")

# List of features in the order your model expects (update as needed)
MODEL_FEATURES = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure'
]

@app.route("/", methods=['GET', 'POST'])
def predict():
    output1 = output2 = ""
    values = {f"query{i}": "" for i in range(1, 20)}
    if request.method == 'POST':
        # Collect input
        data = []
        for i in range(1, 20):
            val = request.form.get(f'query{i}', '')
            values[f"query{i}"] = val
            data.append(val)
        # Build DataFrame
        new_df = pd.DataFrame([data], columns=MODEL_FEATURES)
        # Numeric conversion
        for col in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        # Combine with reference for consistent dummies and binning
        df_2 = pd.concat([df_ref, new_df], ignore_index=True)
        # Tenure binning
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], axis=1, inplace=True)
        # Dummy encoding
        df_dummies = pd.get_dummies(df_2, dtype=int)
        # Align columns with model input
        model_input_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df_dummies.columns
        for col in model_input_cols:
            if col not in df_dummies:
                df_dummies[col] = 0
        df_dummies = df_dummies[model_input_cols]
        # Predict
        single = model.predict(df_dummies.tail(1))[0]
        probability = model.predict_proba(df_dummies.tail(1))[:, 1][0]
        if single == 1:
            output1 = "This customer is likely to be churned!!"
        else:
            output1 = "This customer is likely to continue!!"
        output2 = f"Confidence: {probability * 100:.2f}%"
    return render_template('index.html', output1=output1, output2=output2, **values)

if __name__ == "__main__":
    app.run(debug=True)