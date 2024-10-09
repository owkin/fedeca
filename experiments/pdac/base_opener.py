"""Implement the parent opener used by all centers."""
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# import numpy as np
# import pandas as pd
from scipy.stats import mode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted
from substratools import Opener

# We copy-paste imputation code here to avoid compiling
# from fedeca_dataset.data.transforms.imputation import MissForest


class MissForest(BaseEstimator, TransformerMixin):
    """Missing value imputation using Random Forests.

    MissForest imputes missing values using Random Forests in an iterative
    fashion. By default, the imputer begins imputing missing values of the
    column (which is expected to be a variable) with the smallest number of
    missing values -- let's call this the candidate column.
    The first step involves filling any missing values of the remaining,
    non-candidate, columns with an initial guess, which is the column mean for
    columns representing numerical variables and the column mode for columns
    representing categorical variables. After that, the imputer fits a random
    forest model with the candidate column as the outcome variable and the
    remaining columns as the predictors over all rows where the candidate
    column values are not missing.
    After the fit, the missing rows of the candidate column are
    imputed using the prediction from the fitted Random Forest. The
    rows of the non-candidate columns act as the input data for the fitted
    model.
    Following this, the imputer moves on to the next candidate column with the
    second smallest number of missing values from among the non-candidate
    columns in the first round. The process repeats itself for each column
    with a missing value, possibly over multiple iterations or epochs for
    each column, until the stopping criterion is met.
    The stopping criterion is governed by the "difference" between the imputed
    arrays over successive iterations. For numerical variables (num_vars_),
    the difference is defined as follows:

     sum((X_new[:, num_vars_] - X_old[:, num_vars_]) ** 2) /
     sum((X_new[:, num_vars_]) ** 2)

    For categorical variables(cat_vars_), the difference is defined as follows:

    sum(X_new[:, cat_vars_] != X_old[:, cat_vars_])) / n_cat_missing

    where X_new is the newly imputed array, X_old is the array imputed in the
    previous round, n_cat_missing is the total number of categorical
    values that are missing, and the sum() is performed both across rows
    and columns. Following [1], the stopping criterion is considered to have
    been met when difference between X_new and X_old increases for the first
    time for both types of variables (if available).

    Parameters
    ----------
    NOTE: Most parameter definitions below are taken verbatim from the
    Scikit-Learn documentation at [2] and [3].

    max_iter : int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.

    decreasing : boolean, optional (default = False)
        If set to True, columns are sorted according to decreasing number of
        missing values. In other words, imputation will move from imputing
        columns with the largest number of missing values to columns with
        fewest number of missing values.

    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    criterion : tuple, optional (default = ('mse', 'gini'))
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        NOTE: This parameter is only applicable for Random Forest Classifier
        objects (i.e., for categorical variables).

    Attributes
    ----------
    statistics_ : Dictionary of length two
        The first element is an array with the mean of each numerical feature
        being imputed while the second element is an array of modes of
        categorical features being imputed (if available, otherwise it
        will be None).

    References
    ----------
    * [1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric
      missing value imputation for mixed-type data." Bioinformatics 28.1
      (2011): 112-118.
    * [2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    * [3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from bms_cardio.models.trainer.MissForest import MissForest
    >>> iris = load_iris()
    >>> X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    >>> X.iloc[0, 0] = np.nan
    >>> imputer = MissForest()
    >>> X_imp = imputer.fit_transform(X)
    """

    def __init__(
        self,
        max_iter=10,
        decreasing=False,
        missing_values=np.nan,
        copy=True,
        n_estimators=100,
        criterion=("friedman_mse", "gini"),
        # ("friedman_mse", "gini", "squared_error", "poisson"),
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=1337,
        verbose=0,
        warm_start=False,
        class_weight=None,
    ):
        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.iter_count_ = 0
        self.cat_vars_ = None
        self.num_vars_ = None
        self.statistics_ = None
        self.gamma_new = np.array([0])
        self.gamma_newcat = 0

    def _miss_forest(  # pylint: disable=R0915
        # disable the "too many statement" because refactoring breaks the code
        self,
        Ximp: pd.DataFrame,
        mask: pd.DataFrame,
    ):
        """Run the missForest algorithm."""
        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get("col_means")
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols
            )

            # Reg criterion
            reg_criterion = (
                self.criterion if isinstance(self.criterion, str) else self.criterion[0]
            )
            # Instantiate regression model
            rf_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=reg_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
            )

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get("col_modes")
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(
                col_modes, missing_cat_cols
            )

            # Classfication criterion
            clf_criterion = (
                self.criterion if isinstance(self.criterion, str) else self.criterion[1]
            )

            # Instantiate classification model
            rf_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=clf_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                class_weight=self.class_weight,
            )

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_old = np.inf
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        while (
            self.gamma_new < gamma_old or self.gamma_newcat < gamma_oldcat
        ) and self.iter_count_ < self.max_iter:
            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = self.gamma_new
                gamma_oldcat = self.gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                self.gamma_newcat = (
                    np.sum((Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_]))
                    / n_catmissing
                )
            if self.num_vars_ is not None:
                self.gamma_new = np.sum(
                    (Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2
                ) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            self.iter_count_ += 1

        return Ximp_old

    def fit(
        self,
        X: pd.DataFrame,
        cat_vars: Any = None,  # TODO if we deprecate Python 3.9 change
        # to (int | array of ints | None)
    ):
        """Fit the imputer on X.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            cat_vars needs to be an int or array of int.
        """
        # Check data integrity and calling arguments
        force_all_finite = self.missing_values not in ["NaN", np.nan]

        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = self._get_mask_missforest(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if isinstance(cat_vars, int):
                cat_vars = [cat_vars]
            elif isinstance(cat_vars, (list, np.ndarray)):
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array of ints."
                    )
            else:
                raise ValueError(
                    "cat_vars needs to be either an int or an array of ints."
                )

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else np.array([])

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ["NaN", np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make an initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars.size > 0 else None
        col_modes = (
            mode(X[:, cat_vars], axis=0, nan_policy="omit")[0]
            if cat_vars is not None
            else None
        )

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def transform(self, X_ori: pd.DataFrame):
        """Impute all missing values in X.

        Parameters
        ----------
        X_ori : pd.DataFrame, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The imputed dataset.

        Raises
        ------
        ValueError
            dimension do not match between fitted and data to be transformed.
        """
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = "allow-nan"

        # Create a copy because we need the original for adding colnames
        # to the imputed df

        X_copy = X_ori.copy()

        X = check_array(
            X_copy,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = self._get_mask_missforest(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (
            0
            if self.cat_vars_ is None
            else len([elm for elm in self.cat_vars_ if elm >= 0])
        ) + (
            0
            if self.num_vars_ is None
            else len([elm for elm in self.num_vars_ if elm >= 0])
        )
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError(
                "Incompatible dimension between the fitted "
                "dataset and the one to be transformed."
            )

        # Check if anything is actually missing and if not return original X
        mask = self._get_mask_missforest(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original dataset.")
            return X

        # row_total_missing = mask.sum(axis=1)
        # if not np.any(row_total_missing):
        #     return X

        # Call missForest function to impute missing
        X = self._miss_forest(X, mask)

        # Create a DataFrame from the array with column names
        X_imputed_df = pd.DataFrame(X, columns=X_ori.columns).set_index(X_ori.index)

        # Return imputed dataset
        return X_imputed_df

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y : None
            Not used, present for continuity with scikit-learn.
        fit_params : Any
            Additional fit parameters.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)

    def _get_mask_missforest(self, x: pd.DataFrame, value_to_mask: Any):
        """Compute the boolean mask x == missing_values.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Input data.
        value_to_mask : string or numerical value
            The value to identify as the missing value.

        Returns
        -------
        array-like
            Mask of missing values.
        """
        if value_to_mask == "NaN" or np.isnan(value_to_mask):
            return np.isnan(x)

        return x == value_to_mask


# TODO add metastases location potentially BMI as well
CONT_COLUMNS = [
    "Age at cancer diagnosis",
    "Performance Status (ECOG) at cancer diagnosis",
]
CAT_COLUMNS = ["Biological gender", "Liver metastasis"]
SURV_COLUMNS = ["Overall survival", "Time to death", "Treatment actually received"]

COLUMNS = CONT_COLUMNS + CAT_COLUMNS + SURV_COLUMNS

# TODO compute in a federated fashion
AGE_MAX = 100.0
AGE_MIN = 0.0
# Following inclusion criteria normally we have ECOG 0 to 1 but they are a small
# proportion of ECOG 2s as well
ECOG_MAX = 2.0
ECOG_MIN = 0.0

BMI_MAX = 50.0
BMI_MIN = 10.0


class FedECACenters(str, Enum):
    """Known centers."""

    FFCD = "FFCDMSP"
    IDIBIGI = "IDIBIGi"
    PANCAN = "PanCan"


# Warning !!!!
# The reason for adding a Z as first letter is super arcane but very important
# see substratools inheritance bug line 104 in substratools/utils.py


class ZPDACOpener(Opener):
    """Implement common opener interface.

    Parameters
    ----------
    Opener : _type_
        _description_
    """

    def __init__(self, center_name: FedECACenters):
        """Init the class.

        Parameters
        ----------
        center_name : str
            Name of the center where data is meant to be opened
        """

        self.center_name = FedECACenters(center_name)

    def get_data(self, folders):
        """Open data from a folder.

        Parameters
        ----------
        folders : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert len(folders) == 1, "found more than one folder"
        p = Path(folders[0])
        csv_data_path = p / "data.csv"
        assert csv_data_path.exists(), "data.csv not found in folder"
        # load data
        data = pd.read_csv(csv_data_path)
        assert set(data.columns) == set(
            COLUMNS
        ), f"columns mismatch: {set(data.columns) - set(COLUMNS)}"
        # Very important to preserve column order across centers
        data = data[COLUMNS]
        # =============================================#
        # ==== Common preprocessing across centers ====#
        # =============================================#

        # Survival preprocessing

        data["event"] = data["Time to death"].notnull().astype(int)
        assert set(data["event"].unique()).issubset({0, 1}), "event should be binary"
        assert (
            data["Time to death"] <= 0.0
        ).sum() == 0, "some of the times of death are negative"
        assert np.array_equal(
            data.loc[data["event"] == 1, "Overall survival"].values,
            data.loc[data["event"] == 1, "Time to death"].values,
        ), "Time to death and Overall survival do not coïncide for events"
        assert (
            data["Overall survival"].isnull().sum() == 0
        ), "some of the times of death or censorship are missing"
        data = data.drop(columns=["Time to death"])

        # Treatment preprocessing

        assert set(data["Treatment actually received"].unique()).issubset(
            {"FOLFIRINOX", "Gemcitabine + Nab-Paclitaxel"}
        ), "there should only be the two groups of treatment of interest"
        # FOLFIRINOX being the treatment that is supposedly better we set it to
        # being the treatment
        data["treatment"] = (
            data["Treatment actually received"] == "FOLFIRINOX"
        ).astype(int)
        assert set(data["treatment"].unique()).issubset(
            {0, 1}
        ), "treatment should be binary"
        data = data.drop(columns=["Treatment actually received"])

        # Features preprocessing
        # min-max scaling as in Klein-Brill A, Amar-Farkash S, Lawrence G,
        # Collisson EA, Aran D. Comparison of FOLFIRINOX vs Gemcitabine Plus
        # Nab-Paclitaxel as First-Line Chemotherapy for Metastatic Pancreatic
        # Ductal Adenocarcinoma. JAMA Netw Open.
        # 2022;5(6):e2216199. doi:10.1001/jamanetworkopen.2022.16199
        data["Age at cancer diagnosis"] = [
            (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
            for age in data["Age at cancer diagnosis"].tolist()
        ]
        data["Performance Status (ECOG) at cancer diagnosis"] = [
            (ecog - ECOG_MIN) / (ECOG_MAX - ECOG_MIN)
            for ecog in data["Performance Status (ECOG) at cancer diagnosis"].tolist()
        ]
        # TODO add back BMI if needed
        # data["BMI"] = [
        #    (bmi - BMI_MIN) / (BMI_MAX - BMI_MIN) for bmi in data["BMI"].tolist()
        # ]
        # one hot encoding for categorical variables
        # Assumes all categorical columns are non null
        one_hot_encoder_gender = OneHotEncoder(
            categories=[["M", "F"]], sparse_output=False, drop="first"
        ).set_output(transform="pandas")
        gender_one_hot = one_hot_encoder_gender.fit_transform(
            data[["Biological gender"]]
        )
        data = pd.concat([data, gender_one_hot], axis=1)
        data = data.drop(columns=["Biological gender"])

        one_hot_encoder_liver_meta = OneHotEncoder(
            categories=[[False, True]], sparse_output=False, drop="first"
        ).set_output(transform="pandas")

        liver_meta_one_hot = one_hot_encoder_liver_meta.fit_transform(
            data[["Liver metastasis"]]
        )
        data = pd.concat([data, liver_meta_one_hot], axis=1)
        data = data.drop(columns=["Liver metastasis"])
        # Cat levels are one-hot so they change names + surv columns change names
        # as well

        confounders = CONT_COLUMNS + ["Biological gender_F"] + ["Liver metastasis_True"]
        if data[confounders].isnull().values.any():
            # instantiate imputer
            imputer = MissForest(random_state=42)
            # perform imputation step for counfounders only
            data[confounders] = imputer.fit_transform(data[confounders])

        assert set(data.columns) == set(
            CONT_COLUMNS
            + ["Biological gender_F"]
            + ["Liver metastasis_True"]
            + ["treatment", "event", "Overall survival"]
        )
        assert not data.isnull().values.any(), "there are NaNs in the dataframe"
        # very important last reordering
        data = data[
            CONT_COLUMNS
            + ["Biological gender_F"]
            + ["Liver metastasis_True"]
            + ["treatment", "event", "Overall survival"]
        ]
        data["center"] = self.center_name.value

        # =============================================#
        return data

    def fake_data(self, n_samples):
        """Generate fake data.

        Parameters
        ----------
        n_samples : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        del n_samples
        return pd.DataFrame()
