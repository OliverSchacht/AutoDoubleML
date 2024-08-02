# AutoDoubleML

The python package **AutoDoubleML** provides wrappers for double / debiased machine learning with automated nuisance function hyperparameter optimization. The package is built on top of [**DoubleML**](https://docs.doubleml.org).

See a demo example at [quickstart.ipynb](quickstart.ipynb).

## Installation

Currently, only installation from GitHub is supported.

```{bash}
git clone https://github.com/OliverSchacht/AutoDoubleML
pip install requirements.txt
pip install .
```

## Documentation

For documentation of the double machine learning estimators please refer to the [DoubleML documentation ](https://docs.doubleml.org).

For information about hyperparameter tuning within double machine learning, see our [2024 CLeaR Paper](https://proceedings.mlr.press/v236/bach24a/bach24a.pdf).

**AutoDoubleML** is currently maintained by [@OliverSchacht](https://github.com/OliverSchacht) and [@PhilippBach](https://github.com/PhilippBach).

## Example Usage

```{python}
from autodml.AutoDoubleMLPLR import AutoDoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018
obj_dml_data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20)
adml_plr = adml.AutoDoubleMLPLR(obj_dml_data, time=20)
adml_plr.fit().summary
       coef  std err          t         P>|t|     2.5 %    97.5 %
d  0.485355 0.041147  11.795644  4.110444e-32  0.404708  0.566001
```
## Issues and Contribution

Bugs can be reported to the issue tracker at [https://github.com/OliverSchacht/AutoDoubleML/issues](https://github.com/OliverSchacht/AutoDoubleML/issues).

Contributions from the community are appreciated.