# IPL Match Winner Prediction using Logistic Regression

**An AI mini-project that predicts whether the team batting first will win an IPL match, using information available at the end of the first innings.**

This project is a binary classification task based on IPL match records from 2008–2026. It uses Logistic Regression trained with gradient descent and compares it with scikit-learn's Logistic Regression solver.

> **Prediction timing:** At the innings break. First-innings runs and wickets are known; team-strength and venue-history features are derived only from earlier matches.

---

## Project Overview

The notebook performs:
1. Data loading and exploratory analysis
2. Data cleaning and franchise-name standardization
3. Historical feature engineering
4. Correlation analysis
5. Chronological train/test split
6. Feature scaling and selection
7. Logistic Regression training with gradient descent
8. Evaluation and comparison with baselines
9. Match winner predictions and probability visualization

## Problem Statement

Predict the outcome of an IPL match at the end of the first innings:

- **1:** Team batting first wins
- **0:** Team chasing wins

This is not a pre-match prediction. The first-innings score and wickets lost are available when the prediction is made.

## Dataset

- **Dataset:** IPL matches dataset, 2008–2026
- **File expected by notebook:** `matches.csv`
- **Dataset source:** [IPL 2008–2026 Matches Dataset on Kaggle](https://www.kaggle.com/datasets/arjunsinghgangwar/ipl-20082026-matches-dataset)
- **Target column:** `bat_first_won`

The notebook loads `data/matches.csv` if it exists; otherwise, it looks for `matches.csv` in the repository root.

The notebook standardizes selected renamed franchises, removes matches without a valid winner among the two teams, parses dates, and sorts chronologically. Batting-first and chasing teams are derived from the toss winner and toss decision.

## Features

The notebook builds seven features. Historical values use only matches played before the current match.

| Feature | Description |
|---|---|
| `target_runs` | Runs scored by the team batting first |
| `target_wkts` | Wickets lost by the team batting first |
| `runs_vs_avg` | First-innings runs minus the average first-innings score from the previous 60 matches |
| `elo_diff` | Elo rating of batting-first team minus Elo rating of chasing team |
| `form_diff` | Difference in win rates over the teams' previous five matches |
| `venue_bat1_rate` | Smoothed historical batting-first win share at the venue, centered around 0.5 |
| `bat_first_won_toss` | 1 if the batting-first team won the toss; otherwise 0 |

Elo ratings are updated after each match and carry 60% of their rating into a new season (40% regression toward the mean). The model selects four features using `SelectKBest` with the ANOVA F-test.

## Project Workflow

```text
Kaggle matches.csv
        |
        v
Data cleaning and chronological sorting
        |
        v
Identify batting-first and chasing teams
        |
        v
Historical feature engineering
        |
        v
Correlation analysis
        |
        v
Chronological 80% train / 20% test split
        |
        v
StandardScaler (fit on training data)
        |
        v
Select 4 features (ANOVA F-test)
        |
        v
Logistic Regression with SGD / log-loss
        |
        v
Metrics, plots, and match predictions
```

## Model and Training

- **Model:** Logistic Regression
- **Training implementation:** `SGDClassifier(loss="log_loss")`
- **Regularization:** L2
- **Epochs:** 200
- **Learning rate:** 0.0005 (constant)
- **Alpha:** 0.0001
- **Train/test split:** First 80% chronologically for training; last 20% for testing
- **Scaling:** `StandardScaler`, fitted on training data only
- **Feature selection:** `SelectKBest` with ANOVA F-test, selecting 4 features
- **Random seed:** 42

`partial_fit` is used to record training and testing loss and accuracy over epochs. The notebook also fits scikit-learn's `LogisticRegression` on the selected features for comparison.

## Evaluation

The notebook evaluates train and test data using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log-loss
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Confusion matrix and classification report

### Reported test-set results

The notebook reports results on the most recent 244 matches (2023–2026):

| Metric | Test result |
|---|---:|
| Accuracy | 0.684 |
| Precision | 0.620 |
| Recall | 0.793 |
| F1-score | 0.696 |
| ROC-AUC | 0.770 |
| RMSE | 0.453 |

The notebook also reports a 0.545 majority-class baseline accuracy and 0.676 test accuracy for the exact scikit-learn `LogisticRegression` comparison.

*These are the values documented in the notebook. Results may change if the dataset or notebook is updated and rerun.*

## Visualizations

The notebook produces:

1. Win-margin histograms (wins by runs and wins by wickets)
2. Feature/target correlation heatmap
3. Feature-selection ANOVA F-scores
4. **Training and testing loss vs. epochs**
5. **Training and testing accuracy vs. epochs**
6. Train and test confusion matrices
7. Logistic Regression coefficient plot
8. Batting-first win probability vs. first-innings score

### Expected output screenshot

Final evaluation metrics

<img width="742" height="302" alt="EVALUATION METRICS" src="https://github.com/user-attachments/assets/6a956e31-fd85-499f-886e-768cd7161ab9" />

Prediction Results for the test matches

<img width="1600" height="637" alt="PREDICTION OF A TEST MATCH" src="https://github.com/user-attachments/assets/c8485fc2-2e59-49a7-af70-5f52378d0a8c" />



## Repository Structure

Based on the repository screenshot shared by the team:

```text
ipl-match-winner-prediction/
├── data/
├── .gitignore
├── IPL match prediction.ipynb
├── Readme.md
└── requirements.txt
```

`main.py` is intentionally **not included** in this README's project structure, as the team said it will be removed.

Add the dataset file under `data/` so the notebook can find `data/matches.csv`. If your actual dataset filename differs, update the notebook's `DATA_PATH` or rename the file accordingly.

## Requirements

- Python (recommended: Python 3.10 or later; confirm the version used by your team)
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies from the repository's `requirements.txt`.

## Setup and Run

### 1. Get the repository URL

Open your GitHub repository, click **Code**, and copy the HTTPS URL. It will look like:

```text
https://github.com/USERNAME/REPOSITORY.git
```

Replace the placeholder below with that copied URL.

### 2. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ipl-match-winner-prediction
```

### 3. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Check the dataset location

The notebook checks these locations:

```text
data/matches.csv
matches.csv
```

Download the dataset from the Kaggle link above if it is not committed to the repository, and place it at `data/matches.csv`.

### 6. Run the notebook

```bash
jupyter notebook
```

Open **`IPL match prediction.ipynb`** and select **Run All**.

You can also execute the notebook from the terminal:

```bash
jupyter nbconvert --to notebook --execute "IPL match prediction.ipynb" --output executed_notebook.ipynb
```

Run from the repository root so the relative dataset path resolves correctly.

## Expected Output

After running all notebook cells, the notebook displays:

- Dataset dimensions, result types, missing values, and descriptive statistics
- Data-cleaning checks and target class balance
- Engineered feature values
- Correlation heatmap and feature scores
- Chronological train/test date ranges
- Training and testing loss/accuracy curves
- Train/test evaluation metrics and confusion matrices
- Test-match predicted winners and probabilities
- Sample match predictions
- Probability curve for the batting-first team's chance of winning at different first-innings scores

### Prediction function

The notebook defines `predict_match()` for predictions at the innings break. It takes the batting-first team, chasing team, first-innings runs, and wickets lost.

Example from the notebook's prediction interface:

```python
predict_match(
    "Mumbai Indians",
    "Chennai Super Kings",
    185,
    6
)
```


## Team Members and Roles

| No. | Team member | GitHub username | Role |
|---:|---|---|---|
| 1 | Ankit Bassi | `2026eeb1320` | Documentation Lead |
| 2 | Yatharth Garg | [Add GitHub username] | Documentation Lead |
| 3 | Mannat | `2026aib1055` | Execution Lead |
| 4 | Yash Meena | [Add GitHub username] | PPT Lead |
| 5 | Alankrit Tripathi | [Add GitHub username] | Video Lead |

**Team number:** [Add team number]

All members should contribute to the project and commit from their own GitHub accounts. Roles identify the component lead; they do not replace shared project work.

## Presentation (PPT)


## Project Demo Video

## Limitations and Future Improvements

- Prediction is made at the innings break, not before the match or ball-by-ball during the chase.
- Player-level performance, playing XI, pitch, weather, and dew are not directly modeled.
- The test evaluation covers 244 matches according to the notebook; performance can vary across seasons.
- Historical features and dataset quality affect the predictions.
- Future improvements may include player and playing-XI statistics, pitch/weather data, and ball-by-ball updates during the second innings.

## AI-Use Disclosure

AI tools used by the team: **Claude and ChatGPT**.


- **ChatGPT:** [We used chatgpt ai to assist with project documentation ,including README preparation and improving the clarity of explanation.]
- **Claude:** [We used claude ai to assist with coding and understanding certain parts of the project.AI-generated suggestions were reviewed, modified, and integrated by the team]
  
