import numpy as np


from flask import Flask,render_template,request
import pickle

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])

def predict():
    if request.method=="POST":
        preg=int(request.form["Pregnancies"])
        glu=int(request.form["Glucose"])
        bp=int(request.form["BP"])
        st=int(request.form["SkinThickness"])
        insu=int(request.form["Insulin"])
        bmi=float(request.form["BMI"])
        dpf=float(request.form["DPF"])
        age=int(request.form["Age"])
        
        data=np.array([[preg,glu,bp,st,insu,bmi,dpf,age]])
        my_prediction=classifier.predict(data)
        return(render_template("results.html",prediction=my_prediction))
        
if __name__=="__main__":
    app.run(debug=True)
        