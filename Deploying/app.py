from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('/Users/muhammadzuamaalamin/Documents/KecerdasanBuatan/KecerdasanBuatan/decision_tree_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Output prediction
    output = prediction[0]
    return render_template('index.html', prediction_text='Diabetes Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
