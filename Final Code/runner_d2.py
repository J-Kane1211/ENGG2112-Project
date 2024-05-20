import validate
import modeller
import preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#all current classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


from sklearn.utils.class_weight import compute_sample_weight

filename = "Dataset_2.csv"
df = pd.read_csv(filename, sep=";")


predict = df["cardio"]
DLSR = df[["age", "gender", "height", "weight", "smoke", "alco", "active"]]
GPMC2 = df[["cholesterol", "gluc", "smoke", "alco", "ap_hi", "ap_lo", "age", "gender", "height", "weight", "active"]]

DLSR = preprocess.convert_features(DLSR)
GPMC2 = preprocess.convert_features(GPMC2)

features = [
    DLSR,
    GPMC2
]
feature_names = ["DLSR", "GPMC2"]

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True) #one for each feature set
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
    ax.plot(wnn_fpr, wnn_tpr, label="wnn AUC: {0:.3f}".format(wnn_auc))

    if idx == 0:
        reg_fpr, reg_tpr, reg_auc = modeller.model(features[idx], predict, reg)
        ax.plot(reg_fpr, reg_tpr, label="reg AUC: {0:.3f}".format(reg_auc))

    gnb_fpr, gnb_tpr, gnb_auc = modeller.model(features[idx], predict, gnb)
    ax.plot(gnb_fpr, gnb_tpr, label="gnb AUC: {0:.3f}".format(gnb_auc))

    trw_fpr, trw_tpr, trw_auc = modeller.model(features[idx], predict, trw)
    ax.plot(trw_fpr, trw_tpr, label="trw AUC: {0:.3f}".format(trw_auc))

    ax.set_title(feature_names[idx])
    #individual legend, showing AUC
    ax.legend()


#do some pretty stuff
fig.set_figwidth(12)
fig.supxlabel("fpr")
fig.supylabel("tpr")
fig.savefig('DS2_ROC.png')


