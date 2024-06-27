from flaml import AutoML
from doubleml import DoubleMLIRM
from sklearn.dummy import DummyRegressor, DummyClassifier
import warnings
import numpy as np

class AutoDoubleMLIRM(DoubleMLIRM):
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
                         n_folds=5,
                         n_rep=1,
                         score='ATE',
                         weights=None,
                         normalize_ipw=False,
                         trimming_rule='truncate',
                         trimming_threshold=1e-2,
                         draw_sample_splitting=True)

        if not (isinstance(time, int) or time is None):
            raise TypeError(f'time has to be of type int or None. \
                              {type(time)} was provided.')
        if isinstance(time, int) and time<0:
            raise ValueError(f'time has to be positive. \
                              {time} was provided')
        if time is None:
            time=180
            warnings.warn(f'No optimization time provided. Using default time.')
        if not isinstance(framework, str):
            raise TypeError(f'framework has to be of type string. \
                             {type(framework)} was provided.')
        if not framework in ["flaml"]:
            raise ValueError(f'Currently only framework "flaml" is supported \
                              but {framework} was provided')
        if score=="IV-type":
            raise NotImplementedError('Currently only "partialling out" is supported')
        
        self.time = time

        self.task_g = "classification" if self._dml_data.binary_outcome else "regression"
        if framework == "flaml":
            self.automl_g0 = AutoML(time_budget=self.time / 3,
                                    metric='rmse',  
                                    task=self.task_g)
            self.automl_g1 = AutoML(time_budget=self.time / 3,
                                    metric='rmse',  
                                    task=self.task_g)
            self.automl_m = AutoML(time_budget=self.time / 3,
                                   metric='rmse',  
                                   task="classification")

    def fit(self):
        print(f"Optimizing learners for {self.time}s. Please wait.")
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
        self.set_ml_nuisance_params("ml_g1", "d", self.automl_g1.model.estimator.get_params())
        _fit  = super().fit()
        return _fit