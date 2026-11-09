# IPL Winner Prediction using Logistic Regression

Predict the winner of an IPL match **at the end of the first innings** using first-innings score, wickets, team, and venue information. The notebook trains a Logistic Regression classifier and evaluates it on the newest matches in a time-based test split.

## Problem Statement

The goal is to predict whether the team batting first wins (`1`) or the chasing team wins (`0`) using information available when the first innings ends. Second-innings scores, win margin, and player of the match are not used as prediction features because they are only known after the match.

- **Target:** `bat_first_won` (`1` = batting-first team won, `0` = chasing team won)
- **Model:** Logistic Regression (`C=0.1`)
- **Dataset:** IPL matches dataset (2008–2026), sourced from Kaggle as referenced in the notebook
- **Prediction point:** End of the first innings

## Team Members and Roles

| Team member | Roll/ID | GitHub username | Role |
|---|---|---|---|
| Ankit Bassi | `2026EEB1320` | `2026eeb1320` | Documentation Lead |
| Yatharth Garg | `2026ESB1501` | To be added | Documentation Lead |
| Mannat | `2026AIB1055` | `2026aib1055` | Execution Lead |
| Yash Meena | `2026EEB1426` | To be added | PPT Lead |
| Alankrit Tripathi | `2026MCB1508` | To be added | Video Lead |


## Dataset

The notebook reads `matches.csv`. The dataset contains one row per match and includes match season/date, teams, toss information, venue, innings runs and wickets, winner, and other match metadata.

Notebook dataset summary:
- Original data: **1,243 rows × 31 columns**
- After cleaning: **1,218 matches**
- Training set: **974 matches** (oldest 80%)
- Test set: **244 matches** (newest 20%)

The notebook currently loads the CSV using a local absolute path. For a fresh clone, place the dataset at `data/matches.csv` and update the notebook's `pd.read_csv(...)` line to:

```python
df = pd.read_csv("data/matches.csv")
```

The dataset file must be available locally; it is not included in this repository unless you add it.

## Project Structure

```text
IPL_Winner_Prediction_Added_Only/
├── data/
│   └── matches.csv                 # Dataset (add locally; not included by default)
├── IPL_Winner_Prediction_Added_Only.ipynb
├── requirements.txt
├── README.md
├── slides/
│   └── mini_project_presentation.pptx  # Add the 5-slide presentation
└── video/
    └── project_demo_link.txt           # Add public/unlisted video link
```

## Methodology

1. **Load data** from `matches.csv`.
2. **Clean team names** by merging renamed franchises (Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings, Royal Challengers Bangalore → Royal Challengers Bengaluru, and Rising Pune Supergiants → Rising Pune Supergiant).
3. **Remove matches without a valid winner** and sort matches chronologically.
4. **Identify the batting-first and chasing teams** from the toss winner and toss decision.
5. **Create features:**
   - `target_runs`: batting-first team's first-innings runs
   - `target_wkts`: batting-first team's wickets
   - `runs_vs_avg`: first-innings runs minus the rolling average of the previous 60 matches (using past matches only)
   - batting-first team, chasing team, and venue (one-hot encoded)
6. **Create target:** `bat_first_won`.
7. **Split by time:** oldest 80% for training and newest 20% for testing.
8. **Normalize numeric features** with `StandardScaler`, fitted on training data only.
9. **Select features** using `SelectKBest(f_classif, k=15)`, fitted on training data only.
10. **Train Logistic Regression** and evaluate predictions on the held-out test set.
11. **Train an SGD classifier for 50 epochs** to plot training/testing log loss and create a learning curve.
12. **Predict a new match** using the `predict_ipl_winner()` function.

## Setup

### Requirements

- Python 3.10 or newer (recommended)
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install libraries:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## How to Run

Run the following commands from the repository root, after placing the dataset at `data/matches.csv` and updating the notebook CSV path as described above.

```bash
git clone <YOUR_REPOSITORY_URL>
cd IPL_Winner_Prediction_Added_Only
python -m venv .venv
```

Activate the environment using the command for your operating system, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

1. Open `IPL_Winner_Prediction_Added_Only.ipynb`.
2. Confirm the dataset path points to `data/matches.csv`.
3. Select **Kernel → Restart & Run All**.
4. Run all cells in order to reproduce preprocessing, training, plots, metrics, and the sample prediction.

## Results

The notebook reports the following results on the **244-match test set**:

| Metric | Result |
|---|---:|
| Accuracy | 0.689 |
| Precision | 0.617 |
| Recall | 0.829 |
| F1 score | 0.708 |
| MSE | 0.210 |
| MAE | 0.392 |
| RMSE | 0.459 |
| ROC-AUC | 0.767 |
| Baseline accuracy | 54.5% |

The baseline always predicts the most common target class in the training data. The model's test accuracy is about 13.9 percentage points higher than this baseline.

### Evaluation and Visualizations

The notebook includes:
- Correlation heatmap of numeric features and target
- Training and testing loss versus epochs
- Training and testing accuracy learning curve
- Confusion matrix
- Classification report and ROC-AUC
- Function to predict a winner and win probabilities for a new match

## Example: Predict a New Match

The notebook includes this example:

```python
predict_ipl_winner(
    batting_first_team="Chennai Super Kings",
    chasing_team="Mumbai Indians",
    venue="Wankhede Stadium",
    first_innings_runs=185,
    first_innings_wickets=6
)
```
#Output

Confusion matrix



<img width="795" height="572" alt="confusion_matrix" src="https://github.com/user-attachments/assets/95b2daf5-ddba-4c61-a1bc-7da8d49df5d1" />

Loss vs Epochs



<img width="762" height="562" alt="epochs" src="https://github.com/user-attachments/assets/1421780a-de7c-4d2b-96dd-02386e4d5cbb" />





The function prints the predicted winner and the estimated win probability for both teams. Change the input teams, venue, runs, and wickets to try another match.

**Note:** This is a probability-based estimate, not a guarantee. The prediction uses only information available at the end of the first innings.

## Limitations and Future Work

### Limitations
- Does not use ball-by-ball data, player quality, pitch conditions, or weather.
- Test set contains 244 matches.
- Cricket outcomes have uncertainty, particularly during the chase.
- The notebook's local dataset path must be changed for other computers.

### Future Work
- Add player statistics and more match context.
- Add features from the chase (for example, score after 10 overs) for a later prediction point.
- Experiment with Random Forest or XGBoost and compare performance.

## Project Links

- **GitHub repository:** 
- **Presentation (exactly 5 slides):** 
- **Project video (maximum 5 minutes):** 

## AI-Use Disclosure

- **Claude:** Used to assist with some coding. The team should review and test the generated/assisted code and be able to explain it.
- **ChatGPT:** Used to assist with README/documentation drafting. The team reviewed the documentation against the notebook and project requirements.
- The team is responsible for verifying the code, results, and documentation and for explaining the project during evaluation.
