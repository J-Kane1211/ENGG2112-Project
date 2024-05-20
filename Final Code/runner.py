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


df = pd.read_csv('Dataset_1_combined.csv')

predict = df["HeartDisease"]
feature_DLSR = df[["Age", "Sex", "ChestPainType", "ExerciseAngina"]]
feature_SWSR = df[[
    "Age", "Sex", "ChestPainType",
    "RestingBP", "RestingECG",
    "MaxHR", "ExerciseAngina"
]]
feature_GPMC = df.drop(["Cholesterol", "HeartDisease"], axis=1) #exclude colestrol - feature selection
feature_DLSR = preprocess.convert_features(feature_DLSR)
feature_SWSR = preprocess.convert_features(feature_SWSR)
feature_GPMC = preprocess.convert_features(feature_GPMC)
features = [
    feature_DLSR,
    feature_SWSR,
    feature_GPMC,
]
feature_names = ["DLSR", "SWSR", "GPMC"]

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
    ax.plot(wnn_fpr, wnn_tpr, label="wnn AUC: {0:.3f}".format(wnn_auc))


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
fig.savefig('DS1_ROC.png')


