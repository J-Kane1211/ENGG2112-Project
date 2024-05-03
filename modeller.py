import validate
from sklearn.metrics import roc_curve, auc

def model(features, predict, classifier):
    fpr, tpr = validate.validate_single_classifier(
        features.to_numpy(),
        predict.to_numpy(),
        classifier
    )
    auc_ = auc(fpr, tpr)
    return fpr, tpr, auc_
