import os
import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from flask_cors import cross_origin,CORS
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
@cross_origin()
def index():
    if request.method=="POST":
        try:
            gre_score = float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            research = float(request.form['research']=="yes")

            if(gre_score<0 or gre_score>340):
                return render_template('error.html',type = 'GRE_Score',mm = 0,mx = 340)

            if(toefl_score<0 or toefl_score>120):
                return render_template('error.html', type='TOEFL_Score',mm = 0,mx = 120)
            if(university_rating<1 or university_rating>5):
                return render_template('error.html', type='University_Rating',mm = 1,mx = 5)
            if(sop<1 or sop>5):
                return render_template('error.html', type='SOP',mm = 1,mx = 5)
            if(lor<1 or lor>5):
                return render_template('error.html', type='LOR',mm = 1,mx = 5)
            if(cgpa<0 or cgpa>10):
                return render_template('error.html', type='CGPA',mm = 0,mx = 10)

            x = [[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]]

            load_scaler = pickle.load(open('stdscale.pickle','rb'))
            load_model = pickle.load(open('adimission_chance_reg_model.pickle','rb'))

            x = load_scaler.transform(x)

            chance = max(0,load_model.predict(x)[0])
            chance = min(chance,1)
            print(chance)

            return render_template('predict.html',pred = round(100*chance))

        except Exception as e:
            print(e)
            return render_template('index.html')



if __name__ == "__main__":
    app.run()

