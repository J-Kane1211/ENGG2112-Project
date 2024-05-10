import validate
import modeller
import preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys


#all current classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


from sklearn.utils.class_weight import compute_sample_weight

filename = "Dataset_2.csv"
df = pd.read_csv(filename)


predict = df["cardio"]
featureset_1 = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo"]]
featureset_2 = df[["cholesterol", "gluc", "smoke", "alco" "active", "Sex"]]

featureset_1 = preprocess.convert_features(featureset_1)
featureset_2 = preprocess.convert_features(featureset_2)

features = [
    featureset_1,
    featureset_2
]
feature_names = ["Featureset1", "Featureset2"]

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True) #one for each feature set
for idx, ax in enumerate(axs):
    wnn = KNeighborsClassifier(weights='distance')
    reg = LogisticRegression(max_iter=100000)
    gnb = GaussianNB()
    #this one automatically balances the class labels.
    #this can handle missing values in prediction!
    trw = DecisionTreeClassifier(
        class_weight='balanced',
        min_samples_leaf=5,
        max_depth=8,
        min_weight_fraction_leaf=0.01,
        ccp_alpha=0.002,
    ) 
    

    wnn_fpr, wnn_tpr, wnn_auc = modeller.model(features[idx], predict, wnn)
    ax.plot(wnn_fpr, wnn_tpr, label="wnn AUC: {0:.2f}".format(wnn_auc))


    reg_fpr, reg_tpr, reg_auc = modeller.model(features[idx], predict, reg)
    ax.plot(reg_fpr, reg_tpr, label="reg AUC: {0:.2f}".format(reg_auc))


    gnb_fpr, gnb_tpr, gnb_auc = modeller.model(features[idx], predict, gnb)
    ax.plot(gnb_fpr, gnb_tpr, label="gnb AUC: {0:.2f}".format(gnb_auc))

    trw_fpr, trw_tpr, trw_auc = modeller.model(features[idx], predict, trw)
    ax.plot(trw_fpr, trw_tpr, label="trw AUC: {0:.2f}".format(trw_auc))

    ax.set_title(feature_names[idx])
    #individual legend, showing AUC
    ax.legend()


#do some pretty stuff
fig.set_figwidth(12)
fig.supxlabel("fpr")
fig.supylabel("tpr")
fig.savefig(sys.argv[2])


