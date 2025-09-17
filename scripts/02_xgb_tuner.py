import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from data_prepper import DataPrepper
import catboost as cb
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='data/ts1.csv')
parser.add_argument('--optuna_optimize', type=bool, default=True)
parser.add_argument('-xgb', '--optimize_xgb', action='store_true')
parser.add_argument('-cat', '--optimize_cat', action='store_true')
parser.add_argument('-n','--n_trials', type=int, default=10)
parser.add_argument('--save_cfg', type=bool, default=True)

args = parser.parse_args()
filename = args.filename
optuna_optimize = args.optuna_optimize
optimize_xgb = args.optimize_xgb
optimize_cat = args.optimize_cat
save_cfg = args.save_cfg
n_trials = args.n_trials

class config:

    filename = filename
    # filename = 'data/ts1.csv'
    if filename == 'data/ts1.csv':
        savename = 'ts'
    elif filename == 'data/ehim_full.csv':
        savename = 'ehim'
    elif filename == 'data/whim_full.csv':
        savename = 'whim'
    else:
        savename = 'other'

    dp = DataPrepper(filename)
    train, test = dp.train, dp.test
    full = pd.concat([train, test])
    # tune_train, tune_test = train_test_split(full, test_size=0.37, random_state=42)
    features = ['x', 'y', 'z', 'Area', 'Zmin', 'Zmed', 'Zmax', 'Slope', 'dc_ratio', 'HI', 'sin_Aspect', 'cos_Aspect']
    target = 'target'
    n_rounds = 1
    n_trials = n_trials
    save_cfg = save_cfg

    def save_cfg(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return

def objective_cat(trial):
    n = trial.number
    validation, test_valid = train_test_split(config.test, test_size=0.50)
    params_cat = {
        # 'objective': 'rmse',
        'depth': trial.suggest_int('depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.05, 1.0),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100),
    }
    
    # X_train, y_train = CFG.tune_train[CFG.features], CFG.tune_train[CFG.target] # Random split of 63% train, 37% test
    # X_test, y_test = CFG.tune_test[CFG.features], CFG.tune_test[CFG.target] # Random split of 63% train, 37% test

    train = config.train
    # validation = config.validation
    # test = CFG.test_valid

    X_train, y_train = train[config.features], train[config.target]
    X_valid, y_valid = validation[config.features], validation[config.target]

    cat = cb.CatBoostRegressor(**params_cat, iterations = 100, early_stopping_rounds = 50, bootstrap_type = 'Poisson', task_type ='GPU', silent=True)
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
    y_preds_cat = cat.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, y_preds_cat)
    return rmse

def objective_xgb(trial):
    n = trial.number
    validation, test_valid = train_test_split(config.test, test_size=0.50)

    # Suggest values of the hyperparameters using a trial object.
    params = {
        "objective": 'reg:squarederror',
        'tree_method': "hist",
        "device": "cuda",
        'lambda': trial.suggest_float('lambda', 1e-3, 100.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
    }

    # X_train, y_train = CFG.tune_train[CFG.features], CFG.tune_train[CFG.target] # Random split of 63% train, 37% test
    # X_test, y_test = CFG.tune_test[CFG.features], CFG.tune_test[CFG.target] # Random split of 63% train, 37% test

    train = config.train
    # test = CFG.test_valid

    X_train, y_train = train[config.features], train[config.target]
    X_valid, y_valid = validation[config.features], validation[config.target]
    # X_test, y_test = test[CFG.features], test[CFG.target]

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalidation = xgb.DMatrix(data=X_valid, label=y_valid)
    # dtest = xgb.DMatrix(data=X_test, label=y_test)
    
    model_xgb = xgb.train(
            params,
            dtrain,
            evals=[(dvalidation, 'eval')],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )
    # model = xgb.XGBRegressor(**params)
    # model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_preds_xgb = model_xgb.predict(dvalidation)

    rmse = root_mean_squared_error(y_valid, y_preds_xgb)
    return rmse


def run_xgb_optuna(n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgb, n_trials=n_trials)
    return study, study.best_params, study.best_value

def run_cat_optuna(n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_cat, n_trials=n_trials)
    return study, study.best_params, study.best_value

def get_cat_params(best_params):
    cat_params = {
        # "iterations": 1000,
        "early_stopping_rounds": 50, #100
        'depth': best_params['depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        # 'colsample_bylevel': best_params['colsample_bylevel'],
        'min_data_in_leaf': best_params['min_data_in_leaf'],
        'l2_leaf_reg': best_params['l2_leaf_reg']
    }
    return cat_params

def get_xgb_params(best_params):
    xgb_params = {
        "objective": 'reg:squarederror',
        # "tree_method": "hist",
        # "device": "cuda",
        # "num_boosted_round": 1000, #trial.suggest_int("n_estimators", 1, 2000),
        # "early_stopping_rounds": 50, #100
        'lambda': best_params['lambda'], 
        'alpha': best_params['alpha'], 
        'learning_rate': best_params['learning_rate'], 
        'gamma': best_params['gamma'], 
        'max_depth': best_params['max_depth'], 
        'subsample': best_params['subsample'], 
        'colsample_bytree': best_params['colsample_bytree'], 
        'min_child_weight': best_params['min_child_weight']
    }
    return xgb_params

if __name__ == '__main__':

    print(args)
    # optimize_cat = False
    # optimize_xgb = False
    # print(filename)
    # print(CFG.train.columns)
    
    if optimize_cat and optimize_xgb:
        raise ValueError('Please choose only one model to optimize.')
    if not (optimize_cat or optimize_xgb):
        raise ValueError('Please choose a model to optimize.')
        
    cfg = config()
    
    if optuna_optimize:
        print('Optimizing hyperparameters...')
        if optimize_cat:
            print('Optimizing CatBoost hyperparameters...')
            study, best_params, best_rmse = run_cat_optuna(n_trials)
            cat_params = get_cat_params(best_params)
            config.cat_params = cat_params
            print('-----------------------------------')
            print('Finished optimizing CatBoost hyperparameters.\n')
            if save_cfg:
                cfg.save_cfg(f'cfg.pkl')

        elif optimize_xgb:
            print('Optimizing XGBoost hyperparameters...')
            study, best_params, best_rmse = run_xgb_optuna(n_trials)
            xgb_params = get_xgb_params(best_params)
            config.xgb_params = xgb_params
            print(f'Best XGB Parameters: {best_params}')
            print('-----------------------------------')
            print('Finished optimizing XGBoost hyperparameters.\n')
            if save_cfg:
                cfg.save_cfg(f'cfg.pkl')

