import ThresholdFunction
import validate
import modeller
import preprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys

from sklearn.tree import DecisionTreeClassifier

def main():
    df = pd.read_csv("Dataset_1_combined.csv")
    predict = df["HeartDisease"].to_numpy()
    feature_DLSR_ = df[["Age", "Sex", "ChestPainType", "ExerciseAngina"]]
    feature_SWSR_ = df[[
        "Age", "Sex", "ChestPainType",
        "RestingBP", "RestingECG",
        "MaxHR", "ExerciseAngina"
    ]]
    feature_GPMC_ = df.drop(["Cholesterol", "HeartDisease"], axis=1) #exclude colestrol - feature selection
    
    feature_DLSR = preprocess.convert_features(feature_DLSR_).to_numpy()
    feature_SWSR = preprocess.convert_features(feature_SWSR_).to_numpy()
    feature_GPMC = preprocess.convert_features(feature_GPMC_).to_numpy()

    trw_DLSR = DecisionTreeClassifier(
        class_weight='balanced',
        min_samples_leaf=5,
        max_depth=8,
        min_weight_fraction_leaf=0.01,
        ccp_alpha=0.002,
    )
    trw_SWSR = DecisionTreeClassifier(
        class_weight='balanced',
        min_samples_leaf=5,
        max_depth=8,
        min_weight_fraction_leaf=0.01,
        ccp_alpha=0.002,
    )
    trw_GPMC = DecisionTreeClassifier(
        class_weight='balanced',
        min_samples_leaf=5,
        max_depth=8,
        min_weight_fraction_leaf=0.01,
        ccp_alpha=0.002,
    )
    #train the data now
    for trw, feat in zip(
            (trw_DLSR, trw_SWSR, trw_GPMC),
            (feature_DLSR, feature_SWSR, feature_GPMC)
        ):
        trw.fit(feat, predict)

    while True:
        input_fname = input("Enter file name: ")
        if input_fname == "exit":
            break
        #input_fname = "sample_input.csv"
        feature_df = pd.read_csv(input_fname)

        feature_DLSR = feature_df[["Age", "Sex", "ChestPainType", "ExerciseAngina"]]
        feature_SWSR = feature_df[[
            "Age", "Sex", "ChestPainType",
            "RestingBP", "RestingECG",
            "MaxHR", "ExerciseAngina"
        ]]
        feature_GPMC = feature_df.copy()

        #feature_df.drop(["Cholesterol", "HeartDisease"], axis=1) #exclude colestrol - feature selection
        feature_DLSR = preprocess.convert_features_prod(feature_DLSR, feature_DLSR_).to_numpy()
        feature_SWSR = preprocess.convert_features_prod(feature_SWSR, feature_SWSR_).to_numpy()
        feature_GPMC = preprocess.convert_features_prod(feature_GPMC, feature_GPMC_).to_numpy()

        prob_DLSR = trw_DLSR.predict_proba(feature_DLSR)
        prob_SWSR = trw_SWSR.predict_proba(feature_SWSR)
        prob_GPMC = trw_GPMC.predict_proba(feature_GPMC)

        #print(prob_DLSR, prob_SWSR, prob_GPMC)
        print("DLSR output")
        ThresholdFunction.threshold(prob_DLSR[0][1])
        print("SWSR output")
        ThresholdFunction.threshold(prob_SWSR[0][1])
        print("GPMC output")
        ThresholdFunction.threshold(prob_GPMC[0][1])

if __name__ == "__main__":
    main()
