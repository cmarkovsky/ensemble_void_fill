import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from data_prepper import DataPrepper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from full_tuner import config
from scipy import stats
import xgboost as xgb
import catboost as cb
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='data/ts1.csv')
args = parser.parse_args()
filename = args.filename

class trainer:

    filename = filename
    reg = filename.split('/')[-1].split('_')[0]
    # print('REG!', reg)
    if reg == 'ehimfilt':
        print('Training models on E. Himalaya data.')
        xgb_params = {'objective': 'reg:squarederror',
                      'eval_metric': 'rmse',
                      "tree_method": "hist",
                      "device": "cuda",
                      'lambda': 0.48167593286004784, 
                      'alpha': 0.16112258977667818, 
                      'learning_rate': 0.08190750954630581, 
                      'gamma': 3.9551118720736045, 
                      'max_depth': 18, 
                      'subsample': 0.39347722519998374, 
                      'colsample_bytree': 0.49083564314459427, 
                      'min_child_weight': 1}
        
        cat_params = {'depth': 15, 
                      'learning_rate': 0.06025394575201783, 
                      'subsample': 0.6352006484835869, 
                      'min_data_in_leaf': 62, 
                      'l2_leaf_reg': 7.535667756013887}
        
    elif reg == 'whimfilt':
        print('Training models on W. Himalaya data.')
        xgb_params = {'objective': 'reg:squarederror',
                      'eval_metric': 'rmse',
                      "tree_method": "hist",
                      "device": "cuda",
                      'lambda': 50.70954691133715, 
                      'alpha': 0.04360888728022158, 
                      'learning_rate': 0.037734711537185765, 
                      'gamma': 0.0021677823302037585, 
                      'max_depth': 12, 
                      'subsample': 0.662812768195849, 
                      'colsample_bytree': 0.8861161070600954, 
                      'min_child_weight': 9}
        
        cat_params = {'depth': 15, 
                      'learning_rate': 0.19445698462238312, 
                      'subsample': 0.7685320715978214, 
                      'min_data_in_leaf': 1, 
                      'l2_leaf_reg': 3.6198226518407868}
        
    elif reg == 'combinedfilt':
        print('Training models on combined E. Himalaya and W. Himalaya data.')
        xgb_params = {'objective': 'reg:squarederror',
                      'eval_metric': 'rmse',
                      "tree_method": "hist",
                      "device": "cuda",
                      'lambda': 0.28884257556721954, 
                      'alpha': 3.9402345044638016, 
                      'learning_rate': 0.03312933430551417, 
                      'gamma': 3.2862705612654994, 
                      'max_depth': 20, 
                      'subsample': 0.9608328172848991, 
                      'colsample_bytree': 0.4945528135584324, 
                      'min_child_weight': 78}
        
        cat_params = {'depth': 15, 
                      'learning_rate': 0.16012249670990794, 
                      'subsample': 0.7673709414454766, 
                      'min_data_in_leaf': 2, 
                      'l2_leaf_reg': 1.0015588505196926}
    else:
        print('Region not recognized, default parameters used')
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            "tree_method": "hist",
            "device": "cuda",
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
        }
        cat_params = {
            'depth': 10,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'subsample': 0.5,
            'rsm': 0.5
        }
    dp = DataPrepper(filename)
    train, test = dp.train, dp.test
    full = pd.concat([train, test])
    # tune_train, tune_test = train_test_split(full, test_size=0.37, random_state=42)
    features = ['x', 'y', 'z', 'Area', 'Zmin', 'Zmed', 'Zmax', 'Slope', 'dc_ratio', 'HI', 'sin_Aspect', 'cos_Aspect']
    target = 'target'
        

# def load_cfgs(xgb_filename = f'cfgs/cfg_{reg}_xgb_200.pkl', cat_filename = f'cfgs/cfg_{reg}_cat_200.pkl'):
#     with open(xgb_filename, 'rb') as f:
#         xgb_cfg = pickle.load(f)
#     with open(cat_filename, 'rb') as g:
#         cat_cfg = pickle.load(g)
    
#     return xgb_cfg.xgb_params, cat_cfg.cat_params,


def compute_scores(y, predictions, verbose=False):
    '''returns mae, rmse, mu, med, std, slope, intercept'''
    if np.isnan(predictions).all():
        res = {'mae': np.nan, 'rmse': np.nan, 'mu': np.nan, 'med': np.nan, 'std': np.nan, 'mfit': np.nan, 'qfit': np.nan}

    else:
        # Remove NaNs from both vectors
        mask = ~np.isnan(y) & ~np.isnan(predictions)

        # Filter the vectors
        y = y[mask]
        predictions = predictions[mask]

        mae = mean_absolute_error(y, predictions)
        rmse = root_mean_squared_error(y, predictions)
        mu = np.mean(y - predictions)
        med = np.median(y - predictions)
        std = np.std(y - predictions)
        r2 = r2_score(y, predictions)
        slope, intercept, r_value, p_value, std_err = stats.linregress(y,predictions)
        res = {'mae': mae, 'rmse': rmse, 'mu': mu, 'med': med, 'std': std, 'mfit': slope, 'qfit': intercept}
    if verbose:
        for key in res: print(f"{key}: {res[key]:.2f}", end=", ")
    return tuple(res.values())

def train_models(xgb_params, cat_params):
    models = ['xgb', 'cat']

    
    train = trainer.train
    validation, test= train_test_split(trainer.test, test_size=0.50, random_state = 42)
    preds = []
    # test = trainer.test_valid
    train_preds = {}
    test_preds = {}

    X_train, y_train = train[trainer.features], train[trainer.target]
    X_valid, y_valid = validation[trainer.features], validation[trainer.target]
    X_test, y_test = test[trainer.features], test[trainer.target]
    X_test_full, y_test_full = trainer.test[trainer.features], trainer.test[trainer.target]
    
    if 'xgb' in models:
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalidation = xgb.DMatrix(data=X_valid, label=y_valid)
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        dtest_full = xgb.DMatrix(data=X_test_full, label = y_test_full)
    
        
        print('Training XGBoost model...')
        print('XGBoost parameters:', xgb_params)
        
        model_xgb = xgb.train(
                xgb_params,
                dtrain,
                evals=[(dvalidation, 'eval')],
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=100
            )
    
        y_preds_xgb = model_xgb.predict(dtest)
        preds.append(y_preds_xgb)
        train_preds['xgb'] = model_xgb.predict(dtrain)
        test_preds['xgb'] = model_xgb.predict(dtest_full)

        model_xgb.save_model('xgb_model.json')
        print('XGBoost model trained.')
        print('----------------\n')
        print('XGB Scores: ')
        xgb_scores = compute_scores(y_test, y_preds_xgb, verbose=True)
        print('\n----------------\n')

    if 'cat' in models:
       
        print('Training CatBoost model...')
        print('CatBoost parameters:', cat_params)
        model_cat = cb.CatBoostRegressor(
            **cat_params,
            iterations = 1000,
            early_stopping_rounds = 50,
            loss_function='RMSE',
            verbose=10, 
        )
        
        model_cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
        y_preds_cat = model_cat.predict(X_test)
        preds.append(y_preds_cat)
        train_preds['cat'] = model_cat.predict(X_train)
        test_preds['cat'] = model_cat.predict(X_test_full)
        model_cat.save_model('cat_model.json')
        
        print('CatBoost model trained.')
        print('----------------\n')
        print('Cat Scores: ')
        cat_scores = compute_scores(y_test, y_preds_cat, verbose=True)
        print('\n----------------\n')

    combined_preds = np.mean(preds, axis=0)
    print('Combined scores: ')
    compute_scores(y_test, combined_preds, verbose=True)
    df_train = train.copy()
    df_test = trainer.test.copy() 
    for model in models:
        df_train[model] = train_preds[model]
        df_test[model] = test_preds[model]
        
    df = pd.concat([df_train, df_test])
    
    save_results = True
    if save_results:
        df.to_csv('results.csv', index = False)
    
    


if __name__ == '__main__':    
    print('TRAINING!')
    train_models(trainer.xgb_params, trainer.cat_params)

    