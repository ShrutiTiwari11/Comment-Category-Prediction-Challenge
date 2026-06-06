Comment-Category-Prediction-Challenge
🧠 Comment Category Prediction
Predicting the category assigned to each comment using a mix of text and numerical signals — built an end-to-end ML pipeline with a probability blend ensemble on top.
📊 Best Model: Ensemble (LR + LinearSVC + LightGBM)
🏆 Public Score: 0.83932 | Private Score: 0.82731
📁 Dataset Size: ~198K rows

📌 Problem Statement
Each comment in the dataset belongs to one of 4 categories (0, 1, 2, 3). The task is to predict this category using:

📝 The comment text itself
📊 Numerical signals — votes, emoticons, interaction flags

⚠️ Challenge:

Severe class imbalance — class 0 dominates at ~55%, class 3 is barely 4%
Mixed input types: raw text + structured numbers
Evaluation metric: Macro F1 — every class weighted equally, rare ones included


⚙️ Tech Stack

Python 🐍
pandas, numpy, scipy
scikit-learn
lightgbm
matplotlib, seaborn


🔍 Key Steps
1. 📊 Exploratory Data Analysis (EDA)

Visualized label distribution — imbalance confirmed immediately
All 7 numerical features are heavily right-skewed — log transform required
ANOVA F-test — if_2 has the highest F-statistic, strongest predictor overall; emoticon_1 not significant
IQR outlier analysis — emoticon_1 has the most outliers; if_1, if_2 are cleaner
Class 3 comments are noticeably shorter — text length is a real signal
Dropped race, religion, gender — ~73% missing in both train and test, too sparse to use


2. 🧹 Feature Engineering
🔢 Numerical Features

Log-transformed all skewed columns using np.log1p():
log_upvote, log_downvote, log_if_1, log_if_2, log_emoticon_1/2/3
Interaction features:

upvote_ratio = upvote / (downvote + 1)
upvote_x_if2 = upvote × if_2
emoticon_total = emoticon_1 + emoticon_2 + emoticon_3



✍️ Text Style Features

comm_len, word_count, avg_word_len
upper_ratio — how much of the comment is uppercase
punct_ratio — punctuation density
digit_ratio — digit density
is_intense — binary flag for !!! or ???
hate_count — matches against a custom 15-word toxicity list

📝 Text Processing

Custom cleaner: lowercase → strip HTML → remove URLs/emails → keep alphabets only
Style features extracted before cleaning so punctuation signals aren't lost
Word TF-IDF — 50K features, (1–2)-gram, sublinear_tf=True
Character TF-IDF — 30K features, (3–5)-gram, analyzer='char_wb'
Final matrix: Word + Char TF-IDF + 19 scaled meta features stacked via scipy.sparse.hstack


🤖 Models Used
ModelVal Macro F1RankProbability Blend Ensemble0.8425🥇LightGBM0.8349🥈Logistic Regression0.82🥉LinearSVC (Calibrated)0.794️⃣

🥇 Best Model: Probability Blend Ensemble
Why it worked best:

LR handles linear text patterns, SVC generalizes well on sparse features, LightGBM captures non-linear interactions between text and numerical features — no single model does it all
Grid-searched all weight combinations summing to 1.0
Best weights: LR 0.20 · SVC 0.15 · LGBM 0.65
Added a class-3 alpha boost — multiplied class 3 probabilities by a tuned factor before argmax to push recall up on the rarest class

LightGBM Config:

n_estimators=1200, num_leaves=127, learning_rate=0.03
colsample_bytree=0.7, subsample=0.8, class_weight='balanced'
Early stopping at 80 rounds on multi_logloss


📈 Key Insights

📌 if_2 — strongest predictor across all models, critical for separating class 0
📌 if_1 — most important feature specifically for class 3
📌 Character-level TF-IDF picked up toxic sub-word patterns that word-level missed
📌 Text features lead, but numerical meta features gave a consistent F1 boost
📌 Alpha boosting class 3 probabilities was the final push that improved the score


⚠️ Challenges

Class imbalance — class_weight='balanced' alone wasn't enough for class 3
race, religion, gender had ~73% missing — dropped entirely
LinearSVC needed CalibratedClassifierCV to output probabilities for blending

🚀 Future Improvements

🔥 BERT / RoBERTa for contextual text understanding
🔗 Stacking — train a meta-learner on out-of-fold predictions
🎯 Per-class threshold tuning instead of a single alpha boost
🔁 Stratified K-Fold cross-validation for better generalization
