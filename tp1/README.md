# Deep Learning Applications: Regression and Multiclass Classification

**Student:** Saber Ben Hamda  
**Institution:** Université Abdelmalek Essaâdi — Faculté des Sciences et Techniques de Tanger  
**Program:** Master MBD · **Course:** Deep Learning · **Professor:** Lotfi Elaachak

## Objective

This lab explores how to build, train, tune, and evaluate multilayer perceptrons (MLPs) with PyTorch. It covers a regression task using historical stock prices and a multiclass classification task using machine predictive-maintenance data. Both notebooks include exploratory analysis, hyperparameter search, learning curves, and regularization experiments.

## Repository structure

- [`tp1/part1/01_NYSE_Regression_Colab.ipynb`](tp1/part1/01_NYSE_Regression_Colab.ipynb) — next-trading-day stock-price regression.
- [`tp1/part2/predictive_maintenance_multiclass.ipynb`](tp1/part2/predictive_maintenance_multiclass.ipynb) — multiclass failure-type prediction.

## Getting started

1. Open either notebook in Google Colab:
   - [Open the regression notebook in Colab](https://colab.research.google.com/github/saberBenhamda0/Deep_learning_-_Applications_assignments/blob/main/tp1/part1/01_NYSE_Regression_Colab.ipynb)
   - [Open the classification notebook in Colab](https://colab.research.google.com/github/saberBenhamda0/Deep_learning_-_Applications_assignments/blob/main/tp1/part2/predictive_maintenance_multiclass.ipynb)
2. Run the cells in order. The notebooks install the additional libraries they need and use `kagglehub` to download the datasets. Internet access is required; Kaggle may request authentication. The regression notebook also provides a manual CSV upload fallback.
3. Inspect the generated tables and plots. Training can take several minutes, especially during grid search. A CPU runtime is sufficient.

Main libraries: PyTorch, pandas, NumPy, Matplotlib, seaborn, scikit-learn, imbalanced-learn, skorch, and kagglehub. The regression notebook writes figures, histories, metrics, predictions, and model weights to a `results` directory and offers a ZIP download. Outputs may vary between runs and environments.

## Part 1 — NYSE regression

**Dataset:** [New York Stock Exchange dataset on Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse).

The notebook uses AAPL data from `prices-split-adjusted.csv`. It predicts the next observed trading day's closing price using the current day's open, high, low, close, and volume. The inputs are assumed available after the current day's market close. This is an educational experiment on one stock, not an investment strategy.

### Method

- Explore closing prices, trading volume, daily returns, distributions, and feature correlations.
- Sort observations chronologically and split them into approximately 70% training, 15% validation, and 15% test data. Boundary observations are omitted to keep next-day labels from crossing split boundaries.
- Impute and standardize features using training data, and scale the target for neural-network training.
- Train a PyTorch MLP with ReLU hidden layers and mean squared error loss.
- Use `GridSearchCV` and a skorch wrapper to compare learning rates, Adam and SGD, epoch counts, and hidden-layer sizes. The search uses three time-series cross-validation folds with a one-observation gap and fits preprocessing within each fold.
- Compare a baseline and tuned models with dropout, L2 weight decay, early stopping, and a combined configuration. Select an experiment using validation RMSE.
- Report MAE, RMSE, and R² on training and test data. Compare the MLP with a persistence baseline that predicts tomorrow's close as today's close.

The notebook plots loss and R² against epochs for training and validation, and includes a frozen replay for training/test curves. R² is used instead of classification accuracy because the target is continuous. A widening training–validation gap may indicate overfitting; the plots should be interpreted alongside the metrics.

### Saved results

The saved notebook selected **early stopping** based on validation RMSE. The following values are from its recorded outputs; errors are in the dataset's price units.

| Model | Validation RMSE | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Baseline MLP | 2.103 | 1.154 | 1.603 | 0.954 |
| Early stopping (selected) | 2.014 | 1.095 | 1.528 | 0.958 |
| Persistence baseline | — | 1.063 | 1.504 | 0.960 |

The selected MLP improved validation RMSE compared with the baseline MLP, but the simple persistence baseline had a slightly lower test RMSE. A high R² does not by itself show that the model improves on a simple forecast.

**Grid-search limitation:** the saved regression search output contains PyTorch warnings that prediction and target shapes differ (`[batch, 1]` versus `[batch]`). Broadcasting can make the training loss incorrect, so the recorded grid-search ranking and hyperparameter selection require correction and rerunning before being treated as reliable. The final experiment loop and metrics are reported above as saved, but its chosen configuration depends on that search.

## Part 2 — Predictive-maintenance classification

**Dataset:** [Machine Predictive Maintenance Classification on Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).

The objective is to classify `Failure Type` into six categories, including `No Failure`, from product type and machine measurements. The saved run uses 6,400 fit examples, 1,600 validation examples, and 2,000 test examples.

### Method

- Inspect missing values, duplicate rows, feature distributions, correlations, and class counts.
- Exclude `UDI` and `Product ID` identifiers and the binary `Target` column, which would reveal failure information related to the multiclass label.
- Impute missing values, standardize numerical features, and one-hot encode product type. The preprocessing steps are fitted using the training data.
- Use random oversampling to duplicate minority-class examples in the training data. Within cross-validation, oversampling and preprocessing are fitted separately inside each training fold; validation and test data retain their original class distributions.
- Train a fully connected PyTorch network with ReLU activations and cross-entropy loss. A scikit-learn-compatible wrapper allows it to be used with `GridSearchCV`.
- Compare hidden-layer sizes, learning rates, optimizers, and epoch counts using macro F1. The code uses 16 configurations and two stratified folds. **Note:** a notebook Markdown cell still describes an older, larger search; the code reflects the actual configuration.
- Compare baseline, dropout, L2 weight decay, early stopping, and a combined model using validation macro F1. Plot training/validation loss and accuracy, then evaluate the chosen model on the held-out test set.

### Saved results

The saved grid search reported a best cross-validation macro F1 of approximately **0.549**. The validation comparison selected **early stopping**, with macro F1 of **0.526**, compared with **0.504** for the baseline. The selected model produced the following saved metrics:

| Split | Accuracy | Macro sensitivity / recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Training fit set | 0.9573 | 0.9789 | 0.6684 | 0.9683 |
| Test set | 0.9475 | 0.6114 | 0.5291 | 0.9598 |

Despite high test accuracy, the model achieved **zero recall for Random Failures and Tool Wear Failure** in this run. Those test classes contain only 4 and 9 examples, respectively, so their scores are unstable. The imbalance makes per-class recall and macro F1 especially important; accuracy is dominated by the large `No Failure` class. Random oversampling also duplicates rare examples and can increase overfitting.

The classification notebook plots training and validation learning curves and reports final test metrics, but it does **not** plot test loss and accuracy at every epoch. Validation curves were used for model selection to protect the test set; a frozen replay would be needed to provide the literal test curves requested in the assignment without using them to choose a model.

## Brief synthesis

Through this lab, I learned how to prepare data for regression and multiclass classification, build MLPs using PyTorch, and organize training and evaluation with separate data splits. I used exploratory analysis and preprocessing to understand the datasets, applied grid search to compare hyperparameters, and examined dropout, weight decay, and early stopping. The experiments showed why a time-aware split matters for forecasting and why accuracy alone is misleading for imbalanced classes. They also showed the value of comparing a neural network against a simple baseline and checking warnings before trusting model-selection results.
