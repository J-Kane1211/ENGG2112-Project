import validate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#all current classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("heart.csv")

predict = df["HeartDisease"]
feature_DLSR = df[["Age", "Sex", "ChestPainType", "ExerciseAngina"]]
feature_SWSR = df[[
    "Age", "Sex", "ChestPainType",
    "RestingBP", "RestingECG",
    "MaxHR", "ExerciseAngina"
]]
feature_GPMC = df.loc[:, df.columns != "HeartDisease"]
feature_DLSR = validate.convert_features(feature_DLSR)
feature_SWSR = validate.convert_features(feature_SWSR)
feature_GPMC = validate.convert_features(feature_GPMC)
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
    wnn_fpr, wnn_tpr = validate.validate_single_classifier(
        features[idx].to_numpy(),
        predict.to_numpy(),
        wnn
    )
    wnn_auc = auc(wnn_fpr, wnn_tpr)
    ax.plot(wnn_fpr, wnn_tpr, label="wnn AUC: {0:.2f}".format(wnn_auc))
    reg_fpr, reg_tpr = validate.validate_single_classifier(
        features[idx].to_numpy(),
        predict.to_numpy(),
        reg
    )
    reg_auc = auc(reg_fpr, reg_tpr)
    ax.plot(reg_fpr, reg_tpr, label="reg AUC: {0:.2f}".format(reg_auc))
    gnb_fpr, gnb_tpr = validate.validate_single_classifier(
        features[idx].to_numpy(),
        predict.to_numpy(),
        gnb
    )
    gnb_auc = auc(gnb_fpr, gnb_tpr)
    ax.plot(gnb_fpr, gnb_tpr, label="gnb AUC: {0:.2f}".format(gnb_auc))
    ax.set_title(feature_names[idx])
    #individual legend, showing AUC
    ax.legend()


#do some pretty stuff
fig.set_figwidth(12)
fig.supxlabel("fpr")
fig.supylabel("tpr")
plt.savefig("res.png")


