# Churn Prediction Flask App

This project is a Flask web application that predicts customer churn using a trained Random Forest model. The application allows users to input relevant features and receive predictions on whether a customer is likely to churn.

## Project Structure

```
churn-flask-app
├── app.py                # Main entry point of the Flask application
├── model
│   └── Rf_churn_model.sav # Serialized Random Forest model
├── static
│   └── style.css         # CSS styles for the frontend
├── templates
│   └── index.html        # HTML template for user input and results
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd churn-flask-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application**:
   ```
   python app.py
   ```

2. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:5000`.

3. **Input Features**:
   Fill in the form with the required customer features and submit to get the churn prediction.

## Dependencies

- Flask
- scikit-learn
- pandas
- numpy
- Other dependencies as listed in `requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.