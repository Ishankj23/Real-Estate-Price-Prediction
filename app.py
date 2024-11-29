from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Transaction_Type=request.form.get('Transaction_Type'),
            Registration_type=request.form.get("Registration_type"),
            Area=request.form.get('Area'),
            Property_Type=request.form.get('Property_Type'),
            Property_Sub_Type=request.form.get('Property_Sub_Type'),
            Nearest_Metro=request.form.get('Nearest_Metro'),
            Nearest_Mall=request.form.get('Nearest_Mall'),
            Nearest_Landmark=request.form.get('Nearest_Landamrk'),
            parking=request.form.get('parking'),
            Property_Size=float(request.form.get('Property_Size')),
            Bedrooms=int(request.form.get('Bedrooms'))         
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()

        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True) 