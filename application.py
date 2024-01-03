# Import the dependencies
from flask import Flask, render_template, request
import pickle
import numpy as np

# Create an object for Flask
application = Flask(__name__)

# First let's open pickle file
with open('House_Price.pkl', 'rb') as f:
    model = pickle.load(f)

@application.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@application.route('/predict', methods = ['POST'])
def predict():
    Rooms = int(request.form['bedrooms'])
    Bathrooms = int(request.form['bathrooms'])
    Place = int(request.form['location'])
    Area = int(request.form['area'])
    Status = int(request.form['status'])
    Facing = int(request.form['facing'])
    P_Type = int(request.form['type'])

    # Now take the above data and convert it into array
    input_data = np.array([[Place, Area, Status, Rooms, Bathrooms, Facing, P_Type]])

    # Pass the above data to the model for the prediction
    prediction = model.predict(input_data)[0] # (input_data)[0] --> [0] is placed because it is a 2-d array because sklearn needs 2-d array

    # Now pass the above predicted data to the template
    return render_template('index.html', prediction = prediction)

application.run()
