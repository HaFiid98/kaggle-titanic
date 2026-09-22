import numpy as np


import numpy as np
import pandas as pd


def process_features(df , df_test , age_medians , global_med):
    combined = pd.concat([df.drop(columns = ["Survived"] , errors="ignore") , df_test] ,axis=0).reset_index(drop=True)
    combined["HasCabin"] = combined["Cabin"].notna().astype(int)
    combined["Honorifics"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Mrs_Miss",
        "Mrs": "Mrs_Miss",
        "Master": "Master",
    }
    combined["Honorifics_Median"] = combined["Honorifics"].map(age_medians)
    combined["Age"] = combined["Age"].fillna(combined["Honorifics_Median"])
    combined["Age"] = combined["Age"].fillna(global_med)
    combined["Honorifics"] = combined["Honorifics"].map(title_mapping).fillna("Rare")

    family_count = combined["SibSp"] + combined["Parch"]
    combined["FamilySizeGroup"] = pd.cut(
        family_count,
        bins=[-1, 0, 3, 20],
        labels=["Solo", "Small", "Large"],
    )

    combined["Is_HighStatus_Woman"] = (
        ((combined["Sex"] == "female") & (combined["Pclass"] <= 2)) 
    ).astype(int)


    combined["children_count"] = (
    (combined["Age"].lt(18))
    .groupby(combined["Ticket"])
    .transform("sum")
)


    combined["Age_Pclass"] = combined["Pclass"] * combined["Age"] 
    selected_cols = [
            "Embarked",
            "Age",
            "Age_Pclass" , 
            "HasCabin",
            "Honorifics",
            "Sex",
            "Pclass",
            "FamilySizeGroup",
            "Fare",
            "children_count",
            "Is_HighStatus_Woman",
            
            # "women_children_count"
        ]
        
        
    # print(df)
    train_len = len(df)
    X_train_full = combined.iloc[:train_len].copy()
    X_test_full = combined.iloc[train_len:].copy()
    return X_train_full, X_test_full

