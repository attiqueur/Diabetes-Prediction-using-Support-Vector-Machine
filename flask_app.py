from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])

    # Make prediction
    prediction = model.predict(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])[0]

    # Render the prediction result on the same page
    return render_template('index.html', prediction_result=prediction, pregnancies=pregnancies, glucose=glucose,
                           blood_pressure=blood_pressure,
                           skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,
                           diabetes_pedigree_function=diabetes_pedigree_function, age=age)


if __name__ == '__main__':
    app.run(debug=True)
