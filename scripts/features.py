import numpy as np


def create_features(df):
    df = df.copy()

    df["HasCabin"] = df["Cabin"].notna()

    df["FamilyCount"] = df["Parch"] + df["SibSp"]

    df["FamilySizeGroup"] = np.select(
        [
            df["FamilyCount"] == 0,
            df["FamilyCount"].between(1, 3),
            df["FamilyCount"] > 3
        ],
        [
            "Alone",
            "Small",
            "Large"
        ],
        default="Alone"
    )

    df["Honor"] = df["Name"].str.extract(
        r",\s*([^.]*)\."
    )

    df["Honorifics"] = np.where(
        df["Honor"].isin(["Mr", "Mrs", "Miss", "Master"]),
        df["Honor"],
        "Rare"
    )

    df["Is_HighStatus_Woman"] = (
        (df["Sex"] == "female") &
        (df["Pclass"] <= 2)
    ).astype(int)

    return df