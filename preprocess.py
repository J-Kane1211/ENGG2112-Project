import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import imblearn #try to use downsampling, or use sampling weights


def convert_features(df):
    res = pd.DataFrame()
    enc = OneHotEncoder(drop='first')
    for label in df.columns:
        if df[label].dtype not in ('string', 'category', 'object'):
            res[label] = df[label] #normal add
        else:
            #use one hot encoding for categories and the like
            #get k-1 labels
            new = pd.DataFrame(enc.fit_transform(df[[label]]).toarray(), columns = enc.get_feature_names_out([label]))
            res = pd.concat([
                res,
                new],
                axis=1
            )
    return res
