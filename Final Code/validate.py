import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import sklearn

#calculatees the mean roc
def validate_single_classifier(
    features,
    predict,
    classifier,
    n_splits = 10, #how many validations
):
    model = StratifiedKFold(n_splits=n_splits)
    tpr_all, auc_all, acc_all, recall_all, prec_all, spec_all = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(model.split(features, predict)):
        classifier_fitted = classifier.fit(features[train], predict[train])
        probabilities = classifier_fitted.predict_proba(features[test])
        fpr, tpr, thresholds = roc_curve(predict[test], probabilities[:, 1])
        tpr_all.append(np.interp(mean_fpr, fpr, tpr))
        auc_all.append(auc(fpr, tpr))

        predictions = classifier.predict(features[test])
        acc_all.append(accuracy_score(predict[test], predictions))
        recall_all.append(recall_score(predict[test], predictions))
        prec_all.append(precision_score(predict[test], predictions))
        spec_all.append(recall_score(predict[test], predictions, pos_label=0))

    mean_tpr = np.mean(tpr_all, axis=0)

    print("The accuracy for {} classifier is : {:.3g}".format(classifier, np.mean(acc_all)))
    print("The recall for {} classifier is : {:.3g}".format(classifier, np.mean(recall_all)))
    print("The precision for {} classifier is : {:.3g}".format(classifier, np.mean(prec_all)))
    print("The specificity for {} classifier is : {:.3g}".format(classifier, np.mean(spec_all)))

    try:
        
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        sklearn.tree.plot_tree(classifier, ax=ax)
        fig.savefig("TRW_diagram.png")
    except Exception as e:
        #print(e)
        pass

    return mean_fpr, mean_tpr




if __name__ == "__main__":
    pass




