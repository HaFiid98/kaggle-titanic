## kaggle-titanic

### Overview

The goal of this **1 week** project is to get the highest possible score on a Data Science competition. More precisely you will have to predict who survived the Titanic crash.

![Titanic](titanic.jpg)

#### Kaggle

Kaggle is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges. It's a crowd-sourced platform to attract, nurture, train and challenge data scientists from all around the world to solve data science, machine learning and predictive analytics problems.

#### Titanic - Machine Learning from Disaster

One of the first Kaggle competitions I completed was: Titanic - Machine Learning from Disaster. This is a must-do Kaggle competition.

You can see more [here](https://www.kaggle.com/c/titanic)

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there were not enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

### Role Play

Ahoy, data explorer! Ready to set sail on the most thrilling voyage of your data science career? Welcome aboard the Kaggle Titanic challenge! You're about to embark on a journey through time, back to that fateful night in 1912.
Your mission, should you choose to accept it (and let's face it, you're already hooked), is to dive deep into the passenger manifest and uncover the secrets of survival. Who lived? Who perished? And most importantly, can you build a model that predicts it all?

### Learning Objectives

In this challenge, you have to build a predictive model that answers the question: **"what sorts of people were more likely to survive?"** using passenger data (ie name, age, gender, socio-economic class, etc). **You will have to submit your prediction on Kaggle**.

### Instructions

#### Preliminary

The way the Kaggle platform works is explained in the challenge overview page. If you need more details, I suggest this [resource](https://www.kaggle.com/code/alexisbcook/getting-started-with-kaggle) that gives detailed explanations.

- Create a username following this structure: `username_01EDU_location_MM_YYYY`. Submit the description profile and push it on GitHub the first day of the week. Do not modify this file after the first day.

- It is possible to have different personal accounts merged in a team for one single competition.

#### Scores

In order to validate the project you will have to score at least **78.9% accuracy on the leaderboard**:

- 78.9% accuracy is the minimum score to validate the project.

Scores indication:

- 78.9% difficult - minimum required
- 80% very difficult: smart feature engineering needed
- More than 83%: excellent that corresponds to the top 2% on Kaggle
- More than 85%: cheating

#### Cheating

It is impossible to get 100%.

All people who have 100% of accuracy on the leaderboard cheated, there's no point to compare with them or to cheat. The Kaggle community estimates that having more than 85% is almost considered as cheated submissions as there are elements of luck involved in the surviving.

**You can't use external data sets other than the ones provided in that competition.**

#### The key points

- **Feature engineering**:
  Put yourself in the shoes of an investigator trying to understand what happened exactly in that boat during the crash. Do not hesitate to watch the movie to try to find as many insights as possible. Without smart feature engineering there's no way to pass the project

- The leaderboard evaluates on test data for which you don't have the labels. It means that there's no point to over fit the train set. Check the over fitting on the train set by dividing the data and by cross-validating the accuracy.

### Submission Structure

```console
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

- `README.md` introduction of the project, shows the username, describes the feature engineering and the best score on the **leaderboard**. Note the score on the test set using the exact same pipeline that led to the best score on the leaderboard.

- 'requirements.txt` contains all required libraries to run the code.

- `username.txt` contains the username, the last modified date of the file **has to correspond to the first day of the project**.

- `main.ipynb` This file (single Jupyter Notebook) should contain all steps of data analysis that contributed or not to improve the accuracy, the feature engineering, the model's training and prediction on the test set. It has to be commented to help the reviewers understand the approach and run the code without any bugs.
- **Submit your predictions on the Kaggle's competition platform**. Check your ranking and score in the leaderboard.

### Tips

Don't try to build the perfect model the first day. Iterate a lot and test your assumptions:

Iteration 1:

- Predict all passengers die

Iteration 2

- Fit a logistic regression with a basic feature engineering

Iteration 3:

- Perform an EDA. Make assumptions and check them. Example: What if first class passengers survived more. Check the assumption through EDA and create relevant features to help the model capture the information.

Iteration 4:

- Good luck !

### AI Prompts For Learning

Use these to build the understanding this project needs. They ask the AI to explain concepts and question you, not to write your solution. If the AI starts giving you code, tell it to stop and ask you a question instead.

- "Explain what feature engineering is and why it matters for this competition, then ask me to propose one engineered feature for the Titanic data (such as family size or a title from the name) and justify it."
- "Walk me through how to run an EDA on the Titanic data, then ask me to state a survival hypothesis (for example about class or sex) and predict what the data will show before I check it."
- "Quiz me on what overfitting is and how cross-validation on the train set helps me trust my leaderboard score."
- "Compare logistic regression, decision trees, and random forests for this task, then ask me which I would choose for my best submission and why."
- "Ask me how I handled the missing Age and Cabin values, and push back on the trade-offs of dropping rows versus imputing them."

**A note on using AI:** If AI solves this, AI gets smarter, not you. Every problem you push through yourself is what rewires your brain and builds the skill. Let it think for you and that growth never happens. Use AI to understand, then do the thinking yourself, because your intelligence is the one worth developing.