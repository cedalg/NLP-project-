from flask import Flask,render_template,url_for,request, g
import pandas as pd 
# import pickle
# from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
import itertools as it
import nltk as nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import accuracy_score
# from wordcloud import WordCloud, ImageColorGenerator
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV
# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from keras.models import load_model
from sklearn.svm import SVC
# import pickle 
# import requests
import os
import flask_sijax

path = os.path.join('.', os.path.dirname(__file__), 'static/js/sijax/')

app = Flask(__name__)
app.config['SIJAX_STATIC_PATH'] = path # SIJAX Permet d'acrualisé la page sans la recharger entierement 
app.config['SIJAX_JSON_URI'] = '/static/js/sijax/json2.js'
flask_sijax.Sijax(app)

# filename = 'finalized_model.pkl'
# model = pickle.load(open(filename,'rb'))


@app.route('/')
def home():
	return render_template('home.html',bool_exp="collapse")

# @app.route('/',methods=['POST'])
# def predict():
#     # df = pd.read_csv('C:/Users/yaya/Desktop/Ecole microsoft Data IA/Projet/NLP/Appflask Original/df_prepro_ced.csv')
#     # df = df.drop(['Unnamed: 0'], axis = 1)
#     # X = df['commentaire']
#     # y = df['note']
#     # cv = CountVectorizer()
#     # X = cv.fit_transform(X)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#     #clf = MultinomialNB()
#     # model.fit(X_train,y_train)
#     # model.score(X_test,y_test)
#     # cv = CountVectorizer()
    
#     # if request.method == 'POST':
#     #     message = request.form['message']
#     #     data = [message]
#     #     vect = cv.transform(data).toarray()
#     #     my_prediction = model.predict(vect)
    
#     # return render_template('result.html',prediction = my_prediction)
#     df = pd.read_csv('C:/Users/yaya/Desktop/Ecole microsoft Data IA/Projet/NLP/Appflask Original/df_prepro_ced.csv')
#     df = df.drop(['Unnamed: 0'], axis = 1)
#     X = df['commentaire']
#     y = df['note']
#     cv = CountVectorizer()
#     X = cv.fit_transform(X)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#     clf = MultinomialNB()
#     clf.fit(X_train,y_train)
#     clf.score(X_test,y_test)
    
#     if request.method == 'POST':
#         message = request.form['message']
#         data = [message]
#         vect = cv.transform(data).toarray()
#         my_prediction = clf.predict(vect)
        
#     return {'prediction' : my_prediction}


@flask_sijax.route(app, '/')
def search_sijax():
    def sijax_search_function(obj_response, search_text):
        #obj_response.alert('Search text box value' + search_text)
        message = search_text
        if message:
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = loaded_model.predict(vect)
            
            if my_prediction == 0:
                obj_response.html("#resultat",'<div class="alert alert-success" role="alert">Good review</div>')
            else :
                obj_response.html("#resultat",'<div class="alert alert-danger" role="alert">Bad review</div>')
    
    
    filename = 'finalized_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb')

    df = pd.read_csv('https://raw.githubusercontent.com/lydiahiba/NLP_Project/master/reviews_hotels_preproc.csv')
    df = df.drop(['Unnamed: 0'], axis = 1)
    X = df['commentaire']
    y = df['note']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    clf = SVC()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)

     
    

    
    if g.sijax.is_sijax_request:
        g.sijax.register_callback('sijax_search',sijax_search_function)
        return g.sijax.process_request()
    
    
    
    return render_template('home.html',prediction=my_prediction)

    app.run(debug=True)