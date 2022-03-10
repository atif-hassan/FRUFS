[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/FRUFS.svg)](https://pypi.python.org/pypi/FRUFS/)
[![Downloads](https://pepy.tech/badge/FRUFS)](https://pepy.tech/project/FRUFS)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/FRUFS/commits)
# FRUFS: Feature Relevance based Unsupervised Feature Selection
FRUFS stands for Feature Relevance based Unsupervised Feature Selection and is an unsupervised feature selection technique that uses supervised algorithms such as XGBoost to rank features based on their importance.

## How to install?
```pip install FRUFS```

## Functions and parameters
```python
# The initialization of FRUFS takes in multiple parameters as input
model = FRUFS(model_r, model_c, k, n_jobs, verbose, categorical_features, random_state)
```
- **model_r** - `estimator object, default=DecisionTreeRegressor()` The model which is used to regress current continuous feature given all other features.
- **model_c** - `estimator object, default=DecisionTreeClassifier()` The model which is used to classify current categorical feature given all other features.
- **k** - `float/int, default=1.0` The number of features to select.
	- `float` means to consider `round(total_features*k)` number of features. Values range from `0.0-1.0`
	- `int` means to consider `k` number of features. Values range from `0-total_features`
- **n_jobs** - `int, default=-1` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.
- **verbose** - `int, default=0` Controls the verbosity: the higher, more the messages. A value of 0 displays a nice progress bar.
- **categorical_features** - `list of integers or strings` A list of indices denoting which features are categorical
	- `list of integers` If input data is a numpy matrix then pass a list of integers that denote indices of categorical features
	- `list of strings` If input data is a pandas dataframe then pass a list of strings that denote names of categorical features
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.

```python
# To fit FRUFS on provided dataset and find recommended features
fit(data)
```
- **data** - A pandas dataframe or a numpy matrix upon which feature selection is to be applied\
(Passing pandas dataframe allows using correct column names. Numpy matrix will apply default column names)

```python
# This function prunes the dataset to selected set of features
transform(data)
```
- **data** - A pandas dataframe or a numpy matrix which needs to be pruned\
(Passing pandas dataframe allows using correct column names. Numpy matrix will apply default column names)

```python
# To fit FRUFS on provided dataset and return pruned data
fit_transform(data)
```
- **data** - A pandas dataframe or numpy matrix upon which feature selection is to be applied\
(Passing pandas dataframe allows using correct column names. Numpy matrix will apply default column names)

```python
# To plot XGBoost style feature importance
feature_importance()
```

## How to import?
```python
from FRUFS import FRUFS
```

## Usage
If data is a pandas dataframe
```python
# Import the algorithm. 
from FRUFS import FRUFS
# Initialize the FRUFS object
model_frufs = FRUFS(model_r=LGBMRegressor(random_state=27), model_c=LGBMClassifier(random_state=27, class_weight="balanced"), categorical_features=categorical_features, k=13, n_jobs=-1, verbose=0, random_state=27)
# The fit_transform function is a wrapper for the fit and transform functions, individually.
# The fit function ranks the features and the transform function prunes the dataset to selected set of features
df_train_pruned = model.fit_transform(df_train)
df_test_pruned = model.transform(df_test)
# Get a plot of the feature importance scores
model_frufs.feature_importance()
```

If data is a numpy matrix
```python
# Import the algorithm. 
from FRUFS import FRUFS
# Initialize the FRUFS object
model_frufs = FRUFS(model_r=LGBMRegressor(random_state=27), model_c=LGBMClassifier(random_state=27, class_weight="balanced"), categorical_features=categorical_features, k=13, n_jobs=-1, verbose=0, random_state=27)
# The fit_transform function is a wrapper for the fit and transform functions, individually.
# The fit function ranks the features and the transform function prunes the dataset to selected set of features
X_train_pruned = model.fit_transform(X_train)
X_test_pruned = model.transform(X_test)
# Get a plot of the feature importance scores
model_frufs.feature_importance()
```

## For better accuracy
- Try incorporating more features by increasing the value of k
- Pass strong, hyperparameter-optimized non-linear models

## For better speeds
- Set **n_jobs** to -1

## Performance in terms of Accuracy (classification) and MSE (regression)
| Dataset | # of samples | # of features | Task Type | Score using all features | Score using FRUFS | # of features selected | % of features selected | Tutorial |
| --- | --- | --- |--- |--- |--- |--- |--- |--- |
| Ionosphere | 351 | 34 | Supervised | 88.01 | **91.45** | 24 | 70.5% | [tutorial here](https://github.com/atif-hassan/FRUFS/blob/main/tutorials/ionosphere_supervised-FRUFS.ipynb) |
| Adult | 45222 | 14 | Supervised | 62.16 | **62.65** | 13 | 92.8% | [tutorial here](https://github.com/atif-hassan/FRUFS/blob/main/tutorials/adult_supervised-FRUFS.ipynb) |
| MNIST | 60000 | 784 | Unsupervised | 50.48 | **53.70** | 329 | 42.0% | [tutorial here](https://github.com/atif-hassan/FRUFS/blob/main/tutorials/mnist_unsupervised-FRUFS.ipynb) |
| Waveform | 5000 | 21 | Unsupervised | 38.20 | **39.67** | 15 | 72.0% | [tutorial here](https://github.com/atif-hassan/FRUFS/blob/main/tutorials/waveform_unsupervised-FRUFS.ipynb) |

**Note**: Here, for the first and second task, we use accuracy and f1 score, respectively while for both the fourth and fifth tasks, we use the [NMI metric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html). In all cases, higher scores indicate better performance.

## Future Ideas
- Let me know

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature
