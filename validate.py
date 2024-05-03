import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

#calculatees the mean roc
def validate_single_classifier(
    features,
    predict,
    classifier,
    n_splits = 10, #how many validations
):
    model = StratifiedKFold(n_splits=n_splits)
    tpr_all, auc_all = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(model.split(features, predict)):
        probabilities = classifier.fit(
            features[train], predict[train]).predict_proba(features[test])
        fpr, tpr, thresholds = roc_curve(predict[test], probabilities[:, 1])
        tpr_all.append(np.interp(mean_fpr, fpr, tpr))
        auc_all.append(auc(fpr, tpr))

    mean_tpr = np.mean(tpr_all, axis=0)

    return mean_fpr, mean_tpr


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

if __name__ == "__main__":
    pass




