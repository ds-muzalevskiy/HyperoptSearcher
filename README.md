# HyperoptSearcher

Mini Library with predefined algos and params, one can use for hyperparameter optimisation using Hyperopt

Basic Example:

```
from params import xgb_para
from hyperopt_class import HPOpt, Trials, tpe
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
                           
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

obj = HPOpt(x_train, x_test, y_train, y_test)

xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)

```

