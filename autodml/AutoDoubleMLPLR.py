from flaml import AutoML
from doubleml import DoubleMLPLR
from sklearn.dummy import DummyRegressor
import warnings

from autodml._utils import assert_time

class AutoDoubleMLPLR(DoubleMLPLR):
    def __init__(self,
                 obj_dml_data,
                 framework='flaml',
                 time=None,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         ml_m=DummyRegressor(),
                         ml_l=DummyRegressor(),
                         n_folds=5,
                         n_rep=1,
                         score='partialling out',
                         draw_sample_splitting=True)
        
        time = assert_time(time, self.params_names)
        self.time = time

        self.task_m = "classification" if self._dml_data.binary_treats.all() else "regression"
        if framework == "flaml":
            self.automl_l = AutoML(time_budget=self.time['ml_l'],
                                   metric='rmse',  
                                   task='regression')
            self.automl_m = AutoML(time_budget=self.time['ml_m'],
                                   metric='rmse',  
                                   task=self.task_m)

    def fit(self):
        print(f"Optimizing learners for {self.time}s. Please wait.")
        self.automl_l.fit(X_train=self._dml_data.x, 
                            y_train=self._dml_data.y, verbose=0)
        self.automl_m.fit(X_train=self._dml_data.x, 
                            y_train=self._dml_data.d, verbose=0)
        self._learner = {'ml_l': self.automl_l.model.estimator, 
                            'ml_m': self.automl_m.model.estimator}
        if self.task_m == "classification":
            self._predict_method['ml_m'] = 'predict_proba'
        _fit  = super().fit()
        return _fit