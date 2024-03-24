from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html', title='Diabetes Prediction using Support Vector Machine')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input features from the form
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = int(request.form['age'])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                                    bmi, diabetes_pedigree_function, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Make prediction
        prediction = model.predict(input_data)

        # Map prediction to human-readable label
        if prediction[0] == 0:
            result = 'Non-Diabetic'
        else:
            result = 'Diabetic'

        # Render the template with prediction result included
        return render_template('index.html', title='Diabetes Prediction Result', result=result)


if __name__ == '__main__':
    app.run(debug=True)
