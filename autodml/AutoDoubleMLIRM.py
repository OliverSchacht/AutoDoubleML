from flaml import AutoML
from doubleml import DoubleMLIRM
from sklearn.dummy import DummyRegressor, DummyClassifier
import numpy as np

from autodml._utils import assert_time

class AutoDoubleMLIRM(DoubleMLIRM):
    """Automated double machine learning for interactive regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'ATE'`` or ``'ATTE'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat0, g_hat1, m_hat, smpls)``.
        Default is ``'ATE'``.

    weights : array, dict or None
        An numpy array of weights for each individual observation. If None, then the ``'ATE'`` score
        is applied (corresponds to weights equal to 1). Can only be used with ``score = 'ATE'``.
        An array has to be of shape ``(n,)``, where ``n`` is the number of observations.
        A dictionary can be used to specify weights which depend on the treatment variable.
        In this case, the dictionary has to contain two keys ``weights`` and ``weights_bar``, where the values
        have to be arrays of shape ``(n,)`` and ``(n, n_rep)``.
        Default is ``None``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import autodoubleml as adml
    >>> from doubleml.datasets import make_irm_data
    >>> np.random.seed(3141)
    >>> obj_dml_data = make_irm_data(theta=0.5, n_obs=500, dim_x=20)
    >>> adml_irm_obj = adml.AutoDoubleMLIRM(obj_dml_data, time=30)
    >>> adml_irm_obj.fit().summary
           coef  std err          t         P>|t|     2.5 %    97.5 %
    d  0.598036   0.0497  12.032832  2.388268e-33  0.500625  0.695447
    """
    def __init__(self,
                 obj_dml_data,
                 framework='flaml',
                 time=None,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         ml_m=DummyClassifier(),
                         ml_g=DummyRegressor(),
                         n_folds=n_folds,
                         n_rep=n_rep,
                         score=score,
                         weights=weights,
                         normalize_ipw=normalize_ipw,
                         trimming_rule=trimming_rule,
                         trimming_threshold=trimming_threshold,
                         draw_sample_splitting=draw_sample_splitting)
        
        time = assert_time(time, self.params_names)
        self.time = time

        if score == "ATTE":
            self.time["ml_g0"] += self.time["ml_g1"]
            self.time["ml_g1"] = 0

        if not isinstance(framework, str):
            raise TypeError(f'framework has to be of type string. \
                             {type(framework)} was provided.')
        if not framework in ["flaml"]:
            raise ValueError(f'Currently only framework "flaml" is supported \
                              but {framework} was provided')

        self.task_g = "classification" if self._dml_data.binary_outcome else "regression"

        if framework == "flaml":
            self.automl_g0 = AutoML(time_budget=self.time['ml_g0'],
                                    metric='rmse',  
                                    task=self.task_g)
            self.automl_g1 = AutoML(time_budget=self.time['ml_g1'],
                                    metric='rmse',  
                                    task=self.task_g)
            self.automl_m = AutoML(time_budget=self.time['ml_m'],
                                   metric='rmse',  
                                   task="classification")

    def fit(self, n_jobs_cv=None, store_models=False):
        """
        Estimate AutoDoubleML models. Runs Hyperparameter Optimization for given time and then runs DoubleML.

        Parameters
        ----------
        n_jobs_cv : None or int
            The number of CPUs to use to fit the learners. ``None`` means ``1``.
            Default is ``None``.

        store_models : bool
            Indicates whether the fitted models for the nuisance functions should be stored in ``models``. This allows
            to analyze the fitted models or extract information like variable importance.
            Default is ``False``.

        Returns
        -------
        self : object
        """
                
        learners_info = ', '.join([f"{learner} for {duration}s" for learner, duration in self.time.items()])
        print(f"Optimizing learners: {learners_info}. Please wait.")

        treat_idx = (self._dml_data.d == 1)
        self.automl_g0.fit(X_train=np.c_[self._dml_data.x, self._dml_data.d][~treat_idx],
                           y_train=self._dml_data.y[~treat_idx], verbose=0)
        name_g = self.automl_g0.best_estimator
        self.automl_g1.fit(X_train=np.c_[self._dml_data.x, self._dml_data.d][treat_idx],
                           y_train=self._dml_data.y[treat_idx], estimator_list=[name_g],verbose=0)
        self.automl_m.fit(X_train=self._dml_data.x, 
                          y_train=self._dml_data.d, verbose=0)
        self._learner = {'ml_g': self.automl_g0.model.estimator, 
                         'ml_m': self.automl_m.model.estimator}
        if self.task_g == "classification":
            self._predict_method['ml_g'] = 'predict_proba'
        if self.score == "ATE":
            self.set_ml_nuisance_params("ml_g1", "d", self.automl_g1.model.estimator.get_params())
        _fit  = super().fit(n_jobs_cv=n_jobs_cv, store_models=store_models)
        return _fit