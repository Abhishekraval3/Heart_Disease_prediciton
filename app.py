
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
model_dt = pickle.load(open('model_dt.pkl', 'rb'))
model_knn = pickle.load(open('model_knn.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
model_lr = pickle.load(open('model_ll.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text= '' , total_count='')


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    pr_lr = model_lr.predict(final_features)
    pr_dt = model_dt.predict(final_features)
    pr_knn = model_knn.predict(final_features)
    pr_svm = model_svm.predict(final_features)
    pr_rf = model_rf.predict(final_features)

    prediction = (pr_lr + pr_dt + pr_knn + pr_svm + pr_rf)

    output = prediction/5 
    
    if output>0.5:
        res = "Likely to have a Heart Disease"
    else:
        res = "You are Healthy"
    return render_template('index.html', prediction_text= res )


if __name__ == "__main__":
    app.run(debug=True,port=8088)
    