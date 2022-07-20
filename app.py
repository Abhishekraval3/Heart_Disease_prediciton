
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
model_dt = pickle.load(open('model_dt.pkl', 'rb'))
model_knn = pickle.load(open('model_knn.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
model_lr = pickle.load(open('model_lr.pkl', 'rb'))
model_stack = pickle.load(open('model_stack.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_stack.predict(final_features)

    if prediction == 1:
        res = "Likely to have a cardiovascular Disease"
    else:
        res = "You are Healthy"

   
    return render_template('index.html', prediction_text = res)

if __name__ == "__main__":
    app.run(debug=True,port=8088)
    