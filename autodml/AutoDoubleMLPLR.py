from flaml import AutoML
from doubleml import DoubleMLPLR
from sklearn.dummy import DummyRegressor
import numpy as np
from sklearn.model_selection import cross_val_predict

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
                         ml_g=DummyRegressor() if score=="IV-type" else None,
                         n_folds=n_folds,
                         n_rep=n_rep,
                         score=score,
                         draw_sample_splitting=draw_sample_splitting)
        
        time = assert_time(time, self.params_names)
        self.time = time

        if not isinstance(framework, str):
            raise TypeError(f'framework has to be of type string. \
                             {type(framework)} was provided.')
        if not framework in ["flaml"]:
            raise ValueError(f'Currently only framework "flaml" is supported \
                              but {framework} was provided')
        
        self.task_m = "classification" if self._dml_data.binary_treats.all() else "regression"

        if framework == "flaml":
            self.automl_l = AutoML(time_budget=self.time['ml_l'],
                                   metric='rmse',  
                                   task='regression')
            self.automl_m = AutoML(time_budget=self.time['ml_m'],
                                   metric='rmse',  
                                   task=self.task_m)
            if score=="IV-type":
                self.automl_g = AutoML(time_budget=self.time['ml_g'],
                                    metric='rmse',  
                                    task='regression')

    def fit(self, n_jobs_cv=None, store_models=False):
        learners_info = ', '.join([f"{learner} for {duration}s" for learner, duration in self.time.items()])
        print(f"Optimizing learners: {learners_info}. Please wait.")

        x, y, d = self._dml_data.x, self._dml_data.y, self._dml_data.d
        
        self.automl_l.fit(X_train=x, 
                            y_train=y, verbose=0)
        self.automl_m.fit(X_train=x, 
                            y_train=d, verbose=0)
        self._learner = {'ml_l': self.automl_l.model.estimator, 
                         'ml_m': self.automl_m.model.estimator}
        
        if self.task_m == "classification":
            self._predict_method['ml_m'] = 'predict_proba'
        
        if self._score == "IV-type":
            m_hat = cross_val_predict(self.automl_m.model.estimator, x, d,
                                      method=self._predict_method['ml_m'], cv=self.n_folds, n_jobs=n_jobs_cv)
            l_hat = cross_val_predict(self.automl_l.model.estimator, x, y,
                                      cv=self.n_folds, n_jobs=n_jobs_cv)
            
            psi_a = -np.multiply(d - m_hat, d - m_hat)
            psi_b = np.multiply(d - m_hat, y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

            self.automl_g.fit(X_train=x, 
                              y_train=y - theta_initial*d, verbose=0)
            self._learner["ml_g"] = self.automl_g.model.estimator

        _fit  = super().fit(n_jobs_cv=n_jobs_cv, 
                            store_models=store_models)
        return _fit