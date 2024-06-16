from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_url_path="", static_folder='static')

# Load the trained model
model = joblib.load('../decision_tree_model.joblib')

@app.route('/v2')
def home2():
    return render_template('index_old.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    f = request.form
    for key in f.keys():
        for value in f.getlist(key):
            print (key,":",value)

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Output prediction
    output = prediction[0]
    print("HASIL : " + str(output))
    return jsonify({'prediction': str(output)}
    )
    return render_template('index.html', prediction_text='Diabetes Prediction: {}'.format(output))

@app.route('/predict2', methods=['POST'])
def predict2():
    # Extract data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Output prediction
    output = prediction[0]
    return render_template('index_old.html', prediction_text='Diabetes Prediction: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
