import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from pathlib import Path

from features import process_features
from preprocessing import preprocessor
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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
train,test = process_features(data , test_data , data_medians , global_median)
# test= process_features(test_data ,data   ,  data_medians , global_median)


selected_cols = [
            "Age",
            "Fare",
            "Embarked",
            "HasCabin",
            "Honorifics",
            "Sex",
            "Pclass",
            "FamilySizeGroup",
            "children_count",
            "Is_HighStatus_Woman",
            # "women_children_count"
            "Age_Pclass"
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

X = train[selected_cols]
X_test = test[selected_cols]
print(X_test.isna().sum())

rf_model = RandomForestClassifier(n_estimators=80, random_state=42 , max_leaf_nodes=60)




params = {
    'objective':'binary:logistic',
    'max_depth':8,
    'learning_rate':0.3,
    'n_estimators':100,
    'alpha':10
}

# xg_model = XGBClassifier(**params)
rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=3,  
    min_samples_leaf=5,
    random_state=42,
    
)

# xgb_model = XGBClassifier(
#     n_estimators=100,
#     max_depth=3, 
#     learning_rate=0.03, 
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
# )


X_train,X_test1,y_train,y_test1 = train_test_split(X,y, test_size=0.2, random_state=42 , stratify=y)

# 3. Regularized Logistic Regression
lg_model = LogisticRegression(C=0.1, max_iter=1000)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

model.fit(X=X_train , y=y_train)

trained_rf = model.named_steps["classifier"]
importances = trained_rf.feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(X_test.isna().sum())
plt.figure(figsize=(10, 6))
forest_importances.head(25).plot(kind="bar")
plt.title("Top Feature Importances (MDI)")
plt.ylabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig('training_plots.png')
print("Plot saved as training_plots.png")



# model.feature_importances_

y_predict1 = model.predict(X_test1)
y_predict = model.predict(X_test)
print(f'prediction score : {accuracy_score(y_pred=y_predict1 , y_true=y_test1)}')
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

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf, scoring="accuracy")

print(f"Realistic Local CV Mean: {scores.mean():.4f}")
print(f"Score Variance (Std):   {scores.std():.4f}")