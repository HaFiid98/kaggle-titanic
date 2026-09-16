import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from pathlib import Path

from features import process_features
from preprocessing import preprocessor
import matplotlib.pyplot as plt 
import pandas as pd
# import xgboost as xgb
# from xgboost import XGBClassifier


# train = pd.read_csv("../data/train.csv")
BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path().cwd()

TRAIN_PATH = BASE_DIR.parent / "data" / "train.csv"
TEST_PATH = BASE_DIR.parent / "data" / "test.csv"
# TARGET_PATH = BASE_DIR.parent / "data" / "gender_submission"
data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
y = data["Survived"]
data_medians =data.groupby(
    data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
)["Age"].median()

global_median = data["Age"].median()
train = process_features(data , data_medians , global_median)
test = process_features(test_data  , data_medians , global_median)

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
        "Fare",
        "women_children_count"
    ]
]
X_test = test[
    [
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
]


title_age_medians = train.groupby("Honorifics")["Age"].median()
global_age_median = train["Age"].median()  

train["Age"] = train["Age"].fillna(
    train["Honorifics"].map(title_age_medians)
)
train["Age"] = train["Age"].fillna(global_age_median)

test["Age"] = test["Age"].fillna(
    test["Honorifics"].map(title_age_medians)
)
test["Age"] = test["Age"].fillna(global_age_median)

# rf_model = RandomForestClassifier(n_estimators=100, random_state=42 , max_leaf_nodes=60)
# params = {
#     'objective':'binary:logistic',
#     'max_depth':8,
#     'learning_rate':0.3,
#     'n_estimators':100,
#     'alpha':10
# }

# model = XGBClassifier(**params)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X=X , y=y)
trained_rf = model.named_steps["classifier"]
# importances = trained_rf.feature_importances_
# feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot the top 15 features
# plt.figure(figsize=(10, 6))
# forest_importances.head(25).plot(kind="bar")
# plt.title("Top Feature Importances (MDI)")
# plt.ylabel("Mean Decrease in Impurity")
# plt.tight_layout()
# plt.savefig('training_plots.png')
# print("Plot saved as training_plots.png")


# model.feature_importances_
y_predict = model.predict(X_test)
submission = pd.DataFrame({"PassengerId" : test_data["PassengerId"] , "Survived" :y_predict})
submission.to_csv(BASE_DIR.parent / "data" / "submission.csv" , index=False)
print(y_predict)
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
