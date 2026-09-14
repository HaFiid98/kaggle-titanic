import numpy as np


import numpy as np
import pandas as pd


def process_features(df):
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
    ]
    return df[selected_cols]