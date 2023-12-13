import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("RandomForestClassifier copy.pkl", "rb"))

@flask_app.route("/",  methods = ["GET","POST"] )
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if(prediction==0):
        return render_template("index.html", prediction_text = "Your vehicule is in a good shape Sir, don't worry and have a good drive!")
    else:
        return render_template("index.html", prediction_text = "Your vehicule is in not good shape Sir, Please  check it ")
        
        
    

if __name__ == "__main__":
    flask_app.run(debug=True)