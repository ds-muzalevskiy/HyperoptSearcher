# HyperoptSearcher

Mini Library with predefined algos and params, one can use for hyperparameter optimisation using Hyperopt

To run:

import params
import hyperopt_class

obj = HPOpt(x_train, x_test, y_train, y_test)

xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
