# Création de l'API pour le calcule de prédiction via l'application Web

#Importation des bibliothèques
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, make_scorer
from pydantic import BaseModel
#import joblib

from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score

# Création de l'application
api = FastAPI(
    title="LGBM Prediction",
    description="Predict score",
    version="1.0.0")


#model = Predict()

#Création du modèle
lgbm_model = pickle.load(open('LGBM_best_model.pickle', 'rb'))
X_test = pickle.load(open('X_test_lgbm.pickle', 'rb'))
X_test = X_test.reset_index()
X_test = X_test.drop(['index'], axis=1)
print(X_test)
class ID_client(BaseModel):
    ID : int

#Création du lien
@api.post("/Test")
async def ask_id(id_client:ID_client):
    
    print("ask_id().   ----------------------------------------------------")
    find_client = X_test.iloc[id_client.ID]
    find_client = find_client.array.reshape(1, -1)
    print(find_client)
    probability = lgbm_model.predict_proba(find_client)[:,1] #probabilité
    prediction = lgbm_model.predict(find_client) #prediction
    print("-Probabilité--------------------------------------------------------------------")
    print(probability)
    
    print("-Prédiction---------------------------------------------------------------")
    print(prediction)
    
    print("--------------------------------------------------------------------------")
    
    output = pd.DataFrame({'prediction': prediction, 'probability': probability})
    return output.to_dict(orient = 'records')

@api.get("/test2/{a}/{b}")
async def compute_sum(a, b):
    return {'sum': int(a) + int(b)}
