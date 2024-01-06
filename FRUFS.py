# Feature Relevance based Unsupervised Feature Selection

from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
import contextlib
import joblib
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt


# tqdm enabled for joblib
# This is all thanks to the answer by frenzykryger at https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# Define the class for our unsupervised feature selection algorithm
class FRUFS(TransformerMixin, BaseEstimator):
    def __init__(self, model_r=None, model_c=None, k=1.0, n_jobs=1, verbose=0, categorical_features=None, random_state=None):
        # Define the random state
        self.random_state = random_state
        # By default, the regression model is going to be a decision tree
        if model_r is not None:
            self.model_r = model_r
        else:
            self.model_r = DecisionTreeRegressor(random_state=self.random_state)
        # By default, the classification model is going to be a decision tree
        # Classification is required for categorical variables
        if model_c is not None:
            self.model_c = model_c
        else:
            self.model_c = DecisionTreeClassifier(random_state=self.random_state)
        # Defines the number of features to select. Can be either a fraction between (0,1] or an integer between (0, number of features]
        self.k = k
        # Number of parallel jobs to run.
        self.n_jobs = n_jobs
        # The verbosity level for FRUFS. 0 gives a nice progress bar.
        self.verbose = verbose
        # A list which denotes categorical features if any. Set to None if there are no categorical variables
        self.categorical_features = categorical_features

        # Hidden variables
        self.columns_ = None
        self._feat_imps_ = None


    # This function calculates the importance of all features in predicting the current feature
    def cal_feat_imp_(self, index, X, model):
        # This will contain the feature importance scores
        feat_imp = np.zeros(X.shape[1])
        # Remove the current feature from the dataset
        x_train = np.concatenate((X[:,:index], X[:,index+1:]), axis=1)
        # The current feature is now our target variable
        y_train = X[:,index]
        # Train
        model.fit(x_train, y_train)
        # Generate indices that will be used to add the feature importances in their correct positions
        inds = np.concatenate((np.reshape(np.arange(index), (-1,1)), np.reshape(np.arange(index+1, X.shape[1]), (-1, 1))), axis=0)[:,0]
        # Save the feature importance of each feature
        try:
            feat_imp[inds]+= model.feature_importances_
        except:
            feat_imp[inds]+= model.coef_
        # Return the computed feature importance scores
        return feat_imp


    # The actual feature selection happens here
    def fit(self, X):
        # If pandas dataframe is provided as input then convert to numpy matrix and save column names
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            X = X.values
            # If pandas dataframe is provided as input,
            # then categorical features will exist (if provided) as a list of string containing column names
            # Have to convert that to indices (list of integers)
            if self.categorical_features is not None:
                self.categorical_features = [i for i in range(X.shape[1]) if self.columns_[i] in self.categorical_features]
        else:
            # If numpy matrix is provided as input, then no column names exist
            self.columns_ = np.arange(X.shape[1])
            # Categorical features (if provided) will already be present as indices (list of integers)

        # Instantiate the parallel processing object
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        # If verbosity is set to 0, then show a nice progress bar
        if self.verbose == 0:
            with tqdm_joblib(tqdm(desc="Progress bar", total=X.shape[1])) as progress_bar:
                # Process all feature importance in parallel
                feat_imps = parallel(delayed(self.cal_feat_imp_)(i, X, self.model_r if self.categorical_features is None or i not in self.categorical_features else self.model_c) for i in range(X.shape[1]))
        else:
            # Process all feature importance in parallel
            feat_imps = parallel(delayed(self.cal_feat_imp_)(i, X, self.model_r if self.categorical_features is None or i not in self.categorical_features else self.model_c) for i in range(X.shape[1]))
        # Average out the scores since parallel processing saves all results in the form of a list
        self.feat_imps_ = np.average(np.asarray(feat_imps), axis=0)
        # Remove unnecessary variables
        del feat_imps
        # Sort the columns based on feat_imps_
        inds = np.argsort(np.absolute(self.feat_imps_))[::-1]
        self.columns_ = self.columns_[inds]
        self.feat_imps_ = self.feat_imps_[inds]
        # If top K is provided as a float then take k*(number of features) many features
        if type(self.k) == float:
            self.k = round(self.k * X.shape[1])


    # Prune the dataset
    def transform(self, X):
        # Take only the top k number of features
        cols = self.columns_[:self.k]
        # If data is a pandas dataframe then pruning is different
        if isinstance(X, pd.DataFrame):
            return X[cols]
        else:
            return X[:,cols]


    # Fit and then prune
    def fit_transform(self, X):
        # First run fit
        self.fit(X)
        # Then run transform
        return self.transform(X)


    # Check out the feature importance scores
    def feature_importance(self, top_x_feats=None):
        if top_x_feats is not None:
            y_axis = np.arange(top_x_feats)
            x_axis = self.feat_imps_[:top_x_feats]
        else:
            y_axis = np.arange(len(self.columns_))
            x_axis = self.feat_imps_

        # This dashed line indicates the cut-off point for your features.
        # All features below this line have been pruned
        sns.lineplot(x=x_axis, y=[self.k for i in range(len(y_axis))], linestyle='--')
        # The feature importance scores in the form of a horizontal bar chart
        sns.barplot(x=x_axis, y=y_axis, orient="h")
        # Use feature names if already provided or generate synthetic ones
        if type(self.columns_[0]) == str:
            plt.yticks(y_axis, self.columns_[:len(y_axis)], size='small')
        else:
            plt.yticks(y_axis, ["Feature "+str(i) for i in self.columns_[:len(y_axis)]], size='small')
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.show()
