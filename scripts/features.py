import numpy as np


import numpy as np
import pandas as pd


def process_features(df , age_medians , global_med):
    df = df.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Honorifics"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Mrs": "Mrs_Miss",
        "Miss": "Mrs_Miss",
        "Master": "Master",
    }
    df["Honorifics"] = df["Honorifics"].map(title_mapping).fillna("Rare")

    family_count = df["SibSp"] + df["Parch"]
    df["FamilySizeGroup"] = pd.cut(
        family_count,
        bins=[-1, 0, 3, 20],
        labels=["Solo", "Small", "Large"],
    )

    df["Is_HighStatus_Woman"] = (
        (df["Sex"] == "female") & (df["Pclass"] <= 2)
    ).astype(int)


    df["women_children_count"] = (
    (df["Sex"].eq("female") | df["Age"].lt(18))
    .groupby(df["Ticket"])
    .transform("sum")
)
    selected_cols = [
            "Embarked",
            "Age",
            "HasCabin",
            "Honorifics",
            "Sex",
            "Pclass",
            "FamilySizeGroup",
            "Is_HighStatus_Woman",
            "Fare",
            "women_children_count"
        ]
        
        
    # print(df)
    df["Honorifics_Median"] = df["Honorifics"].map(age_medians)
    df["Age"] = df["Age"].fillna(df["Honorifics_Median"])
    df["Age"] = df["Age"].fillna(global_med)
    return df[selected_cols]
