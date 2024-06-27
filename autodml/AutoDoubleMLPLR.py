from flaml import AutoML
from doubleml import DoubleMLPLR
from sklearn.dummy import DummyRegressor
import warnings

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

        if not (isinstance(time, int) or time is None):
            raise TypeError(f'time has to be of type int or None. \
                              {type(time)} was provided.')
        if isinstance(time, int) and time<0:
            raise ValueError(f'time has to be positive. \
                              {time} was provided')
        if time is None:
            time=120
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

        self.task_m = "classification" if self._dml_data.binary_treats.all() else "regression"
        if framework == "flaml":
            self.automl_l = AutoML(time_budget=self.time / 2,
                                   metric='rmse',  
                                   task='regression')
            self.automl_m = AutoML(time_budget=self.time / 2,
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