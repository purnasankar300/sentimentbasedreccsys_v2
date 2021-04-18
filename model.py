import pandas as pd
import pickle

# TF IDF
vectorizer = pickle.load(open("tdidf_fit_vector.pickel", "rb"))
# MINMAX scalling
normalize = pickle.load(open("scalling_transform.pickel", "rb"))
# Random Forest model
random_forest_model = pickle.load(open("finalized_model_randomforest.pickel", "rb"))
# User-User recomendation system
recc_df = pickle.load(open("recc_sys_cosine_corr.pickle", "rb"))
# Text preprocess review data
review_df=pd.read_csv('sample30_textpreprocess_data.csv',index_col='name')


def result_predict(product):
    return_list=[]
    dataframe_df=review_df[review_df.index==product]
    dataframe_reviews = vectorizer.transform(dataframe_df['review_clean_data']) 
    dataframe_reviews_norm=normalize.transform(dataframe_reviews.toarray()) 
    target_predit=random_forest_model.predict(dataframe_reviews_norm) 
    postivper=round(((target_predit==1).sum()*100 )/len(target_predit),2)
    negativper=round(((target_predit==0).sum()*100)/len(target_predit),2)
    return postivper,negativper