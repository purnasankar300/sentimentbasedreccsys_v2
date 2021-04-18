from flask import Flask,render_template,request
import pickle
import pandas as pd
from model import result_predict

recc_df = pickle.load(open("recc_sys_cosine_corr.pickle", "rb"))

app = Flask(__name__)


@app.route("/",methods =["POST","GET"])
def home():
    if request.method == "POST":
        user_id = request.form.get("userid")
        user_id=user_id.lower().strip()
        if len(user_id)==0:
            return render_template('base.html') + 'PLEASE ENTER USER ID'
        if user_id not in recc_df.index:
            return render_template('base.html') + 'THE USER ID IS NOT AVILABLE IN DATASET PLEASE USE VALID USER ID'
        else:  
            product_name=recc_df.loc[user_id].sort_values(ascending=False)[0:20].index.tolist()
            result_df=pd.DataFrame(columns=['Product','Positive%','Negative%'])
            for prod in product_name:
                postivper,negativper=result_predict(prod)
                result_df = result_df.append({'Product':prod,'Positive%':postivper,'Negative%':negativper},ignore_index = True)
            result_df.sort_values(by=['Positive%'], inplace=True,ascending=False)
            return render_template('home.html',predict=result_df.head(5),user=user_id) 

    else:
        return render_template('base.html')  
    

if __name__ == "__main__":
    app.run(debug=True)