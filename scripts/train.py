import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from pathlib import Path

from features import create_features
from preprocessing import preprocessor


train = pd.read_csv("../data/train.csv")
BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path().cwd()

TRAIN_PATH = BASE_DIR.parent / "data" / "train.csv"
TEST_PATH = BASE_DIR.parent / "data" / "test.csv"
TARGET_PATH = BASE_DIR.parent / "data" / "gender_submission"
train = create_features(TARGET_PATH)
test = create_features(TEST_PATH)
X = train[
    [
        "Embarked",
        "Age",
        "HasCabin",
        "Honorifics",
        "Sex",
        "Pclass",
        "FamilySizeGroup",
        "Is_HighStatus_Woman",
        "Fare"
    ]
]

y = train["Survived"]


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("Scores:", scores)
print("Mean:", scores.mean())