import os
import sys
sys.path.append('./')
sys.path.append('./data/')
sys.path.append('./models/')

import warnings
warnings.filterwarnings('ignore')

import shutil
import argparse
import importlib
from tqdm import tqdm
from pytz import timezone
from datetime import datetime

import wandb
import json 
from torch.cuda import amp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import gc
from tqdm import tqdm_notebook
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold

from data.common import *
from sklearn.metrics import log_loss
import random 
import lightgbm as lgb
import sklearn.metrics as metrics

def predict_each_fold(cfg, train_df, valid_df, test_df, is_feat_eng=True, params=None):
    print('predict_each_fold', cfg.label_col)
    if is_feat_eng:
        features, train_df, valid_df, test_df = get_features(
            cfg, 
            train_df,
            valid_df,
            test_df
        )
    else:
        features = [
            c for c in train_df.columns if c not in [
                'is_clicked', 'is_installed', 'file_name', 'f_0', 'f_1', 'f_7', 'f_7_count_full', 'f_9', 'f_11', 'f_43', 'f_51', 'f_58', 'f_59', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70'
                ]  + cfg.delete_features
            ]
            
    # get dataloader
    trn_lgb_data, val_lgb_data = get_dataloader(
        cfg, 
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        features=features,
        label_col=cfg.label_col
        )
    
    
    # get model 
    evals = {}
    params["random_state"] = cfg.seed
    clf = lgb.train(
        params, trn_lgb_data, cfg.num_iterations, valid_sets = [trn_lgb_data, val_lgb_data], 
        categorical_feature = cfg.categorical_features, 
        callbacks = [
            lgb.early_stopping(stopping_rounds=100), 
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals)
            ]
    )
    
    preds = clf.predict(valid_df[features])
    
    logs = {
        'Train Loss': clf.best_score['training']['binary_logloss'],
        'Valid Loss': clf.best_score['valid_1']['binary_logloss'],
        'Valid Metric': normalized_cross_entropy(valid_df[cfg.label_col], preds)
    }

    return clf, preds, logs, evals, train_df, valid_df, test_df, features

def main():
    # parse arguments
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.device}'
    cfg = importlib.import_module(f'configs.{args.config}').cfg
    cfg.update(args)
    # init everything
    init_everything(cfg)
    
    print('verbose', cfg.verbose)

    if cfg.verbose:
        msg = "Arguments\n"
        for k, v in vars(cfg).items():
            msg += f"  - {k}: {v}\n"
        print(msg)
    
    # seed everything
    seed_everything(cfg.seed)

    # load raw data
    dfs = load_data(cfg)

    # preprocess raw data
    (
        test_df, 
        train_df
        ) = preprocess(cfg, *dfs)

    '''
    Original 데이터 사용할 경우 
    '''
    columns = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9',
       'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
       'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27',
       'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36',
       'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_43', 'f_44', 'f_45',
       'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54',
       'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_60', 'f_61', 'f_62', 'f_63',
       'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', 'f_71', 'f_72',
       'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79']
    train_df = train_df[columns + ['is_clicked', 'is_installed']]
    test_df = test_df[columns]

    params = cfg.get("params")
    print(params)

    # train/valid split
    if cfg.split == "time":
        (
            train_df,
            valid_df,
            ) = train_valid_split(
                cfg, train_df)
        
        clf, _, logs, evals, train_df, valid_df, test_df, features = predict_each_fold(cfg, train_df, valid_df, test_df, is_feat_eng=True, params=params)
            
        ax = lgb.plot_importance(clf, max_num_features=50, figsize=(20, 20))    
        ax.figure.savefig(f'{cfg.save_dir}/{cfg.name}/feature_importance.png', dpi=300)
        
        preds = clf.predict(valid_df[features])
        
        logs = {
            'Train Loss': clf.best_score['training']['binary_logloss'],
            'Valid Loss': clf.best_score['valid_1']['binary_logloss'],
            'Valid Metric': normalized_cross_entropy(valid_df[cfg.label_col], preds)
        }            
        
        subs = clf.predict(test_df[features].reset_index(drop=True))
        score = logs['Valid Loss']

    else: 
        trn_idx = train_df[train_df["f_1"] != 66].index
        val_idx = train_df[train_df["f_1"] == 66].index
        valid_df = train_df.iloc[val_idx]
        train_df = train_df.iloc[trn_idx]

        features, train_df, valid_df, test_df = get_features(
            cfg, 
            train_df,
            valid_df,
            test_df
        )

        i = 0
        num_folds = 22
        subs = np.zeros(test_df.shape[0])
        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
        oofs = np.zeros(train_df.shape[0])
        folds = train_valid_split(cfg, train_df)
        for i, (trn_idx, val_idx) in enumerate(folds):
            i += 1
            print(f"FOLD {i}")
            _train_df = train_df[features + ["is_installed"]].iloc[trn_idx]
            _valid_df = train_df[features + ["is_installed"]].iloc[val_idx]
            print(_train_df.shape, _valid_df.shape)
            
            clf, _, logs, evals, _, _, test_df, features = predict_each_fold(cfg, _train_df, _valid_df, test_df, is_feat_eng=False, params=cfg.params)
            tmp_preds = clf.predict(test_df[features].reset_index(drop=True))
            
            subs += tmp_preds / num_folds
            oofs[val_idx] = clf.predict(_valid_df[features])

        train_df["oofs"] = oofs
        valid_df = train_df[train_df["f_1"] == 66].reset_index(drop=True)
        score = metrics.log_loss(valid_df.is_installed, valid_df.oofs, labels=[0,1], eps=1e-7, normalize=True)

    if not cfg.no_wandb:
        wandb.log(logs)

    print(features)
    # submission 
    score = logs['Valid Loss']

    submission = pd.DataFrame() 
    submission["row_id"] = test_df["f_0"]
    submission["is_clicked"] = subs
    submission["is_installed"] = subs
    save_file = 'submission' + f'(cfg_{cfg.name})_cv{score}_split{cfg.split}.csv'
    submission.to_csv(f'{cfg.save_dir}/{cfg.name}/{save_file}', sep ='\t', index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="recsys challenge 2023")
    parser.add_argument(
        '--config', required=True,
        help="config filename")
    parser.add_argument(
        '--seed', type=int, default=47, 
        help="seed (default: 47)")
    parser.add_argument(
        '--no_wandb', action='store_true', default=False,
        help="no wandb (default: False)")
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help="verbose or not (default: False)")
    parser.add_argument(
        '--device', default='0', 
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu') 
    parser.add_argument(
        '--smoothing', type=float, default=0.1
    )
    args = parser.parse_args()
    return args

def init_everything(cfg):
    """ Init everything
    - init environment 
    - init project
    - init logging
    Args:
        cfg: config file
    Returns:
        None
    """   
    # init project
    version = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M")
    cfg.project = f'{cfg.user_name}.{cfg.project}'
    cfg.name = f'{cfg.config}.seed{cfg.seed}'
    
    print(cfg.api_key)
    # init logging
    if not cfg.no_wandb:
        wandb.login(key=cfg.api_key)
        wandb.init(
            project=cfg.project,
            name=cfg.name,
            config=cfg.__dict__,
            job_type='Train',
            )
    os.makedirs(f'{cfg.save_dir}/{cfg.name}', exist_ok=True)
    shutil.copy(f'configs/{cfg.config}.py', f'{cfg.save_dir}/{cfg.name}/config.py')

def seed_everything(seed=42):
    """Seed everything
    Args:
        seed: seed number
    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def normalized_cross_entropy(y_true, y_pred): 
    N = len(y_true)
    p = sum(y_true) / N
     
    result = 0
    '''
    for true, pred in zip(y_true, y_pred):
        result += (true * np.log(pred) + (1-true) * np.log(1-pred))
    '''
    result = log_loss(y_true, y_pred, normalize=False)
    return -(1/N)*result*(1/(p*np.log(p) + (1-p)*(np.log(1-p))))

if __name__ == '__main__':
    main()