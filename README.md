# Spanish La Liga & Top 5 Leagues Match Predictor

## 1. Problem Statement

The goal of this project is to predict the outcome of football matches from the Top 5 European Leagues (La Liga, Premier League, Bundesliga, Serie A, Ligue 1) using historical match statistics. This is a multi-class classification problem where the target variable `FTR` (Full Time Result) has three possible values:

- **H**: Home Win
- **D**: Draw
- **A**: Away Win

By analyzing features such as shots, shots on target, fouls, corners, and cards, we aim to build machine learning models that can accurately classify the match result.

## 2. Dataset Description

- **Source**: [football-data.co.uk](https://www.football-data.co.uk/data.php)
- **Content**: Combined match data from **Top 5 European Leagues** (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) for the last 3 seasons (2021-2024).
- **Instances**: > 5000 matches.
- **Features Used**: Full set of **12 features**
  - `HS` (Home Team Shots), `AS` (Away Team Shots)
  - `HST` (Home Shots on Target), `AST` (Away Shots on Target)
  - `HF` (Home Fouls), `AF` (Away Fouls)
  - `HC` (Home Corners), `AC` (Away Corners)
  - `HY` (Home Yellow Cards), `AY` (Away Yellow Cards)
  - `HR` (Home Red Cards), `AR` (Away Red Cards)

## 3. Models Used & Comparison Table

We implemented and evaluated 6 classification models on the combined dataset using the full feature set (12 features).

| ML Model Name       | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| :------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.5957 | 0.7521 |     0.572 | 0.5957 | 0.5567 | 0.3566 |
| Decision Tree       |   0.4616 | 0.5812 |    0.4675 | 0.4616 |  0.463 | 0.1697 |
| kNN                 |   0.5051 | 0.6471 |    0.5067 | 0.5051 | 0.5014 | 0.2377 |
| Naive Bayes         |   0.5523 | 0.6996 |    0.5201 | 0.5523 | 0.4975 | 0.2782 |
| Random Forest       |   0.5837 | 0.7194 |    0.5598 | 0.5837 | 0.5553 | 0.3367 |
| XGBoost             |   0.5449 | 0.6999 |    0.5196 | 0.5449 | 0.5276 | 0.2755 |

## 4. Observations

- **Logistic Regression** remains the most robust model on this dataset with an accuracy of **59.57%**, slightly improving over the 4-feature model (59.11%). This suggests that while Shots on Target and Corners are dominant, the other 8 features provide some marginal gain.
- **Random Forest** showed improved performance (**58.37%**) compared to the reduced feature set (51.71%), indicating it effectively utilizes the additional features like Fouls and Cards to make better splits.
- **Decision Tree** continues to perform poorly due to high variance.
- **Complexity Trade-off**: The gain in accuracy (~0.4%) from using all 12 features versus just 4 is minimal for the best model (Logistic Regression), but significant for ensemble models like Random Forest.

## 5. How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run the training script (optional, models are pre-saved): `python train_models.py`
3. Run the Streamlit app: `streamlit run app.py`
