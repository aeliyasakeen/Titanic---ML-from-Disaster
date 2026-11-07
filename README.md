# Titanic: Machine Learning from Disaster

A beginner-friendly solution to the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). The goal is to predict which passengers survived the Titanic shipwreck using passenger information such as age, sex and ticket class.

Built in Google Colab with Python, pandas and scikit-learn.

## Results

| Metric | Score |
|---|---|
| Hold-out validation accuracy (80/20 split) | ~0.79 |
| 5-fold cross-validation accuracy | ~0.81 |
| Kaggle public leaderboard score | _add your score here_ |

## Dataset

Data comes from the Kaggle competition:

- `train.csv`: 891 passengers with the outcome (`Survived`)
- `test.csv`: 418 passengers without the outcome, used for the final predictions

Main columns: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.

## Approach

1. **Explore the data.** Check shape, columns and missing values.
2. **Clean the data.**
   - Filled missing `Age` and `Fare` with the median from the training data
   - Filled missing `Embarked` with the most common port
   - Dropped `Cabin` (about 77% missing) and `Ticket`, `Name`, `PassengerId` (unique to each passenger, so no general pattern to learn)
3. **Encode text as numbers.**
   - `Sex`: male = 0, female = 1
   - `Embarked`: one-hot encoded
4. **Split the data.** 80% for training, 20% for validation.
5. **Train a model.** Random Forest classifier with 100 trees.
6. **Evaluate.** Accuracy on the validation set, plus 5-fold cross-validation for a more stable estimate.
7. **Predict.** Apply the same cleaning to `test.csv`, train on all 891 passengers, and write `submission.csv`.

The same cleaning steps are applied to train and test, using values computed from the training data only.

## Findings

- The number of trees (100, 150, 300) made almost no difference in cross-validation.
- Adding a `Title` feature extracted from `Name` did not noticeably improve the score, since it largely repeats information already in `Sex` and `Age`.

## How to run

1. Open the notebook in Google Colab.
2. Add your Kaggle API token to Colab Secrets (key icon in the left sidebar) and turn on notebook access.
3. Join the competition on Kaggle and accept the rules.
4. Run all cells. The notebook downloads the data with `kagglehub` and creates `submission.csv`.

```python
import kagglehub
path = kagglehub.competition_download('titanic')
```

Never commit your Kaggle API token to GitHub.

## Requirements

- Python 3
- pandas
- scikit-learn
- kagglehub

```bash
pip install pandas scikit-learn kagglehub
```

## Possible improvements

- Feature engineering such as family size and whether a passenger travelled alone
- Hyperparameter tuning with grid search
- Trying other models such as gradient boosting or logistic regression

## Acknowledgements

Dataset and competition by [Kaggle](https://www.kaggle.com/competitions/titanic).
