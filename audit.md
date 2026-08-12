#### General

##### Submission Structure

```
project
│   README.md
│   requirements.txt or (environment.yml)
│   username.txt
│
└───data
│   │   train.csv
│   |   test.csv
|   |   gender_submission.csv
│
└───notebook
│   │   main.ipynb
|
|───scripts
│

```

###### Does the structure of the project look like above?

##### Play the Role of a Stakeholder

The auditor plays the role of a teammate reviewing the learner's work before it ships. Spend 5-10 minutes asking the learner to explain their own pipeline. The goal is to confirm the learner produced their leaderboard score themselves and understands it, not just that a number appears. A learner who did the work can answer these; one who only copied a notebook cannot.

1. Walk me through your feature engineering: which features you created and why.
2. How did you check for overfitting, and what did cross-validation on the train set tell you?
3. Why is a leaderboard score above 85% on this competition considered suspicious?
4. Which model did you choose for your best submission, and why over the alternatives?
5. Pick a passenger profile, predict whether your model says survived or not, then run it and compare.

###### Can the learner explain their feature engineering and model choice without reading from the code?

###### Can the learner correctly predict their model's output for a passenger profile given live during the audit?

###### Does the learner demonstrate genuine understanding of the pipeline that produced their leaderboard score, rather than reciting generated code?

##### Documentation and Setup

###### Does the `README.md` give an introduction to the project and show the username?

###### Does the `README.md` describe the feature engineering used?

###### Does the `README.md` show the best score obtained on the leaderboard?

###### Does the environment contain all libraries used and their versions that are necessary to run the code?

##### Feature engineering

###### Can the notebook be executed without any error?

###### Does the notebook explain the feature engineering that contributed to improving the accuracy?

##### Model training and prediction

###### Can you train the best model on the train data with feature engineering (via the notebook or the scripts) without any error?

###### Can you predict on the test set using the best model without any error?

###### Is the score on the test set with the best model close to what is expected?

##### Final score

###### Is the accuracy associated with the username in `username.txt` at least 78.9% (the best submission score can be accessed from the learner's Kaggle profile)?
