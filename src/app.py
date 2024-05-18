# https://www.datacamp.com/tutorial/machine-learning-models-api-python
# https://medium.com/red-buffer/how-to-build-a-rest-api-for-your-machine-learning-model-using-flask-8c2fbc75e359

# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model_loaded:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(model_loaded.predict(query))
            return jsonify({'prediction': str(prediction)})
            
            # df = pd.DataFrame(json_)
            # prediction = model_loaded.predict(df)
            # return jsonify({"Prediction": list(prediction)})
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    # Load the model file.
    model_loaded = joblib.load("./model/iris_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("./model/iris_model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
