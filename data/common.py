import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import numpy as np 
from itertools import combinations
import category_encoders as ce
import gc
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from collections import defaultdict 

def downcast_df(df):
    float_columns = df.select_dtypes('float').columns
    int_columns = df.select_dtypes('int').columns
    df[float_columns] = df[float_columns].apply(pd.to_numeric, downcast='float')
    df[int_columns] = df[int_columns].apply(pd.to_numeric, downcast='integer')
    return df
    
def load_data(cfg):
    """ Load raw data without any processing
    """
    test_df = None
    train_df = None

    train_df = pd.read_parquet(cfg.train_file)
    if cfg.test_file.endswith('.parquet'):
        test_df = pd.read_parquet(cfg.test_file)
    else:
        test_df = pd.read_csv(cfg.test_file, sep='\t')
    
    if cfg.downcast:
        train_df = downcast_df(train_df)
        test_df = downcast_df(test_df)
        
    rets = (
        test_df,
        train_df
        )
    
    return rets

def preprocess(
    cfg,
    test_df, 
    train_df,
    ):
    """ Preprocess raw data
    """        
    if cfg.get('filtering', False):
        def remove_feature(data): 
            data.drop(["f_24", "f_25"], axis=1, inplace=True) # f23~f25 = f23
            data.drop(["f_27", "f_28", "f_29"], axis=1, inplace=True) # f26~f29 = f26
            data.drop(["f_7"], axis=1, inplace=True) # cardinarity 1 
            return data 

        train_df = remove_feature(train_df)
        test_df = remove_feature(test_df)
    
    # 빠른 실험 위한 세팅 
    if cfg.get('sub_training', False): 
        # 66제외 1주일만 추출 
        train_df = train_df[train_df["f_1"].isin([55,56,57,58,59,60,61,62,63,64,65,66])].reset_index(drop=True)

    rets = (
        test_df, 
        train_df
        )
    
    return rets

def train_valid_split(
    cfg, train_df # kfold, last
    ):
    print(f'train_valid_split: {cfg.split}')
    if cfg.split == "time":
        trn_idx = train_df[train_df["f_1"] != 66].index
        val_idx = train_df[train_df["f_1"] == 66].index
        valid_df = train_df.iloc[val_idx]
        train_df = train_df.iloc[trn_idx]
        
        rets = (
            train_df,
            valid_df
        )
        return rets
    
    elif cfg.split == "gkfold":
        gkf = GroupKFold(n_splits=22)
        folds = gkf.split(train_df, train_df["is_installed"], groups=train_df['f_1'])
        
        return folds
    elif cfg.split == "skfold":
        skf = StratifiedKFold(n_splits=22, shuffle=True, random_state=42)
        folds = skf.split(train_df, train_df["is_installed"])
        
        return folds
    
    elif cfg.split == "tscv":
        train_df = train_df.sort_values(by=["f_1", "f_0"])
        tscv = TimeSeriesSplit(n_splits=22)
        folds = tscv.split(train_df)
        
        return folds
    else:
        raise NotImplementedError(f"split: {cfg.split}")
    
def get_features(
    cfg,
    train_df=None,
    valid_df=None,
    test_df=None,
): 
    if cfg.get('feat_f_others', False): 
        for col in tqdm([51, 64, 65, 66, 67, 68, 69, 70]): 
            col = f"f_{col}"
            v1 = sorted(train_df[col].unique())[1]
            v2 = sorted(train_df["f_42"].unique())[1]
            train_df[f"feat_{col}"] = np.around(train_df["f_42"]/v2)/(1+np.around(train_df[col]/v1))
            valid_df[f"feat_{col}"] = np.around(valid_df["f_42"]/v2)/(1+np.around(valid_df[col]/v1))
            test_df[f"feat_{col}"] = np.around(test_df["f_42"]/v2)/(1+np.around(test_df[col]/v1))

    if cfg.get('feat_na', False):
        train_df["f_43_is_null"] = train_df["f_43"].isnull().astype(int)
        valid_df["f_43_is_null"] = valid_df["f_43"].isnull().astype(int)
        test_df["f_43_is_null"] = test_df["f_43"].isnull().astype(int)     

        train_df["f_64_is_null"] = train_df["f_64"].isnull().astype(int)
        valid_df["f_64_is_null"] = valid_df["f_64"].isnull().astype(int)
        test_df["f_64_is_null"] = test_df["f_64"].isnull().astype(int)     

    # 6.296107 to 6.466871
    if cfg.get('feat_count_preprocessing', False):
        for col in tqdm([42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]): 
            col = f"f_{col}"
            v = sorted(train_df[col].unique())
            train_df[col] = np.around(train_df[col]/v[1])
            valid_df[col] = np.around(valid_df[col]/v[1])
            test_df[col] = np.around(test_df[col]/v[1])

    if cfg.get('feat_log_continuous', False):
        for col in tqdm([42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]): 
            col = f"f_{col}"
            train_df[col] = train_df[col].apply(lambda x: np.log1p(x))
            valid_df[col] = valid_df[col].apply(lambda x: np.log1p(x))
            test_df[col] = test_df[col].apply(lambda x: np.log1p(x))  

    if cfg.get('feat_f_51', False): 
        train_df["feat_f_51"] = (train_df["f_43"]/(train_df["f_51"]+1))
        valid_df["feat_f_51"] = (valid_df["f_43"]/(valid_df["f_51"]+1))
        test_df["feat_f_51"] = (test_df["f_43"]/(test_df["f_51"]+1))

    if cfg.get('feat_f_56', False): 
        train_df["feat_f_56"] = (train_df["f_51"]/(train_df["f_56"]+1))
        valid_df["feat_f_56"] = (valid_df["f_51"]/(valid_df["f_56"]+1))
        test_df["feat_f_56"] = (test_df["f_51"]/(test_df["f_56"]+1))

    if cfg.get('feat_f_55', False): 
        train_df["feat_f_55"] = (train_df["f_55"]/(train_df["f_57"]+1))
        valid_df["feat_f_55"] = (valid_df["f_55"]/(valid_df["f_57"]+1))
        test_df["feat_f_55"] = (test_df["f_55"]/(test_df["f_57"]+1))

    if cfg.get('feat_f_61', False): 
        train_df["feat_f_61"] = (train_df["f_61"]/(train_df["f_63"]+1))
        valid_df["feat_f_61"] = (valid_df["f_61"]/(valid_df["f_63"]+1))
        test_df["feat_f_61"] = (test_df["f_61"]/(test_df["f_63"]+1))

    if cfg.get('feat_f_66', False): 
        train_df["feat_f_66"] = (train_df["f_66"]/(train_df["f_70"]+1))
        valid_df["feat_f_66"] = (valid_df["f_66"]/(valid_df["f_70"]+1))
        test_df["feat_f_66"] = (test_df["f_66"]/(test_df["f_70"]+1))
    
    # 6.296107 to 6.229302
    if cfg.get('feat_frequency_encoding', False):
        # 저장해둔 피쳐 파일이 있으면 해당 파일을 불러와서 concat 하기 
        # train_feat.parquet, valid_feat.parquet, test_feat.parquet 
        frequency_path = "/data/project/recsys-challenge-2023/sharechat_recsys2023_data_submit/feat/frequency"
        if not os.path.isfile(os.path.join(frequency_path, 'fe_trv2.parquet')):
            cols = [f"f_{c}" for c in [2,3,4,5,6,8,12,13,14,15,16,17,18,19,20,21,22,23,24,32]] # 9, 11넣고도 해보기 # if c not in [9, 11]
            for col in tqdm(cols, position=1, leave=False):
                for t in range(45, 65): # 전체 days에 대해서, f_2 - f_33
                    fe = train_df[train_df["f_1"] == t][col].value_counts(dropna=False)
                    train_df.loc[train_df["f_1"] == t+1, f'{col}_count_full'] = train_df.loc[train_df["f_1"] == t+1, col].map(fe) / (train_df[train_df["f_1"] == t].shape[0])

                fe = train_df[train_df["f_1"] == 65][col].value_counts(dropna=False)
                valid_df.loc[:, f'{col}_count_full'] = valid_df.loc[:, col].map(fe) / (train_df[train_df["f_1"] == 65].shape[0])
                
                fe = valid_df[col].value_counts(dropna=False)
                test_df.loc[:, f'{col}_count_full'] = test_df.loc[:, col].map(fe) / (valid_df.shape[0])
            gc.collect()
        else:
            tr_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_trv2.parquet'))
            val_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_valv2.parquet'))
            te_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_tev2.parquet'))
            columns = ['f_0',
                        'f_2_count_full',
                        'f_3_count_full',
                        'f_4_count_full',
                        'f_5_count_full',
                        'f_6_count_full',
                        'f_8_count_full',
                        'f_12_count_full',
                        'f_13_count_full',
                        'f_14_count_full',
                        'f_15_count_full',
                        'f_16_count_full',
                        'f_17_count_full',
                        'f_18_count_full',
                        'f_19_count_full',
                        'f_20_count_full',
                        'f_21_count_full',
                        'f_22_count_full',
                        'f_23_count_full',
                        'f_24_count_full',
                        'f_32_count_full',
            ]

            train_df = pd.merge(train_df, tr_fe[columns], how='left', on=["f_0"])
            valid_df = pd.merge(valid_df, val_fe[columns], how='left', on=["f_0"])
            test_df = pd.merge(test_df, te_fe[columns], how='left', on=["f_0"])

    if cfg.get('feat_f_43', False):
        def interval_to_label(x): 
            for i in range(10): 
                if x > categories.categories[i].left and x <= categories.categories[i].right: 
                    return i 

        # Assign labels based on the intervals : 64~70
        for col in ["f_43"]: 
            categories = pd.qcut(train_df[col].values, 10)

            valid_df[col] = pd.cut(valid_df[col], bins=categories.categories)
            test_df[col] = pd.cut(test_df[col], bins=categories.categories)
            train_df[col] = pd.cut(train_df[col], bins=categories.categories)

            valid_df[f"feat_{col}"] = list(valid_df[col].apply(lambda x: interval_to_label(x.right)).values)
            test_df[f"feat_{col}"] = list(test_df[col].apply(lambda x: interval_to_label(x.right)).values)
            train_df[f"feat_{col}"] = list(train_df[col].apply(lambda x: interval_to_label(x.right)).values)

    if cfg.get('feat_frequency_encoding_7days', False):
        frequency_path = "/data/project/recsys-challenge-2023/sharechat_recsys2023_data_submit/feat/frequency_7days"
        if not os.path.isfile(os.path.join(frequency_path, 'fe_trv2.parquet')):
            cols = [f"f_{c}" for c in [2, 4, 6, 13, 15, 18]] # 9, 11넣고도 해보기 # if c not in [9, 11]
            for col in tqdm(cols, position=1, leave=False):
                for t in range(45, 65): # 전체 days에 대해서, f_2 - f_33
                    temp = train_df[(train_df["f_1"] > t-7) & (train_df["f_1"] <= t)]
                    fe = temp[col].value_counts(dropna=False)
                    train_df.loc[train_df["f_1"] == t+1, f'{col}_count_full_7days'] = train_df.loc[train_df["f_1"] == t+1, col].map(fe) / (temp.shape[0])

                t = 65
                temp = train_df[(train_df["f_1"] > t-7) & (train_df["f_1"] <= t)]
                fe = temp[col].value_counts(dropna=False)
                valid_df.loc[:, f'{col}_count_full_7days'] = valid_df.loc[:, col].map(fe) / (temp.shape[0])
                
                t = 66
                temp = pd.concat([train_df[(train_df["f_1"] > t-7) & (train_df["f_1"] <= t)], valid_df], axis=0)
                fe = temp[col].value_counts(dropna=False) # 이거 먼저 오류 수정하고 내보기 
                test_df.loc[:, f'{col}_count_full_7days'] = test_df.loc[:, col].map(fe) / (temp.shape[0])

            os.makedirs(frequency_path, exist_ok=True)
            train_df[["f_0"] + [f"f_{c}_count_full_7days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_trv2.parquet'))
            valid_df[["f_0"] + [f"f_{c}_count_full_7days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_valv2.parquet'))
            test_df[["f_0"] + [f"f_{c}_count_full_7days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_tev2.parquet'))
        else:
            tr_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_trv2.parquet'))
            val_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_valv2.parquet'))
            te_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_tev2.parquet'))

            train_df = pd.merge(train_df, tr_fe, how='left', on=["f_0"])
            valid_df = pd.merge(valid_df, val_fe, how='left', on=["f_0"])
            test_df = pd.merge(test_df, te_fe, how='left', on=["f_0"])

    if cfg.get('feat_frequency_encoding_full_days', False):
        frequency_path = "/data/project/recsys-challenge-2023/sharechat_recsys2023_data_submit/feat/frequency_full_days"
        os.makedirs(frequency_path, exist_ok=True)
        if not os.path.isfile(os.path.join(frequency_path, 'fe_tr.parquet')):
            cols = [f"f_{c}" for c in [2, 4, 6, 13, 15, 18]] # 9, 11넣고도 해보기 # if c not in [9, 11]
            for col in tqdm(cols, position=1, leave=False):
                for t in range(45, 65): # 전체 days에 대해서, f_2 - f_33
                    temp = train_df[(train_df["f_1"] <= t)]
                    fe = temp[col].value_counts(dropna=False)
                    train_df.loc[train_df["f_1"] == t+1, f'{col}_count_full_days'] = train_df.loc[train_df["f_1"] == t+1, col].map(fe) / (temp.shape[0])

                t = 65
                temp = train_df
                fe = temp[col].value_counts(dropna=False)
                valid_df.loc[:, f'{col}_count_full_days'] = valid_df.loc[:, col].map(fe) / (temp.shape[0])
                
                t = 66
                temp = pd.concat([train_df, valid_df], axis=0)
                fe = temp[col].value_counts(dropna=False) # 이거 먼저 오류 수정하고 내보기 
                test_df.loc[:, f'{col}_count_full_days'] = test_df.loc[:, col].map(fe) / (temp.shape[0])

            os.makedirs(frequency_path, exist_ok=True)
            train_df[["f_0"] + [f"f_{c}_count_full_days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_tr.parquet'))
            valid_df[["f_0"] + [f"f_{c}_count_full_days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_val.parquet'))
            test_df[["f_0"] + [f"f_{c}_count_full_days" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_te.parquet'))
        else:
            tr_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_tr.parquet'))
            val_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_val.parquet'))
            te_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_te.parquet'))

            train_df = pd.merge(train_df, tr_fe, how='left', on=["f_0"])
            valid_df = pd.merge(valid_df, val_fe, how='left', on=["f_0"])
            test_df = pd.merge(test_df, te_fe, how='left', on=["f_0"])

    if cfg.get('feat_frequency_encoding_full_daysv2', False):
        frequency_path = "/data/project/recsys-challenge-2023/sharechat_recsys2023_data_submit/feat/frequency_full_daysv2"
        os.makedirs(frequency_path, exist_ok=True)
        if not os.path.isfile(os.path.join(frequency_path, 'fe_tr.parquet')):
            cols = [f"f_{c}" for c in [2, 4, 6, 13, 15, 18]] # 9, 11넣고도 해보기 # if c not in [9, 11]
            for col in tqdm(cols, position=1, leave=False):
                for t in range(45, 65): # 전체 days에 대해서, f_2 - f_33
                    temp = train_df[(train_df["f_1"] == t)]
                    fe = temp[col].value_counts(dropna=False)
                    train_df.loc[train_df["f_1"] == t+1, f'{col}_count_full_daysv2'] = train_df.loc[train_df["f_1"] == t+1, col].map(fe) / (temp.shape[0])

                t = 65
                temp = train_df
                fe = temp[col].value_counts(dropna=False)
                valid_df.loc[:, f'{col}_count_full_daysv2'] = valid_df.loc[:, col].map(fe) / (temp.shape[0])
                
                t = 66
                temp = pd.concat([train_df, valid_df], axis=0)
                fe = temp[col].value_counts(dropna=False) # 이거 먼저 오류 수정하고 내보기 
                test_df.loc[:, f'{col}_count_full_daysv2'] = test_df.loc[:, col].map(fe) / (temp.shape[0])

            os.makedirs(frequency_path, exist_ok=True)
            train_df[["f_0"] + [f"f_{c}_count_full_daysv2" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_tr.parquet'))
            valid_df[["f_0"] + [f"f_{c}_count_full_daysv2" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_val.parquet'))
            test_df[["f_0"] + [f"f_{c}_count_full_daysv2" for c in [2, 4, 6, 13, 15, 18]]].to_parquet(os.path.join(frequency_path, 'fe_te.parquet'))
        else:
            tr_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_tr.parquet'))
            val_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_val.parquet'))
            te_fe = pd.read_parquet(os.path.join(frequency_path, 'fe_te.parquet'))

            train_df = pd.merge(train_df, tr_fe, how='left', on=["f_0"])
            valid_df = pd.merge(valid_df, val_fe, how='left', on=["f_0"])
            test_df = pd.merge(test_df, te_fe, how='left', on=["f_0"])

    if cfg.get('feat_frequency_encoding_full_daysv3', False):
        all_df = pd.concat([train_df, valid_df], axis=0)
        cols = [f"f_{c}" for c in [2,3,4,5,6,8,12,13,14,15,16,17,18,19,20,21,22,23,24,32]] # 9, 11넣고도 해보기 # if c not in [9, 11]
        for col in cols: 
            fe = all_df[col].value_counts()
            train_df[f"feat_frequency_{col}"] = train_df[col].map(fe)
            valid_df[f"feat_frequency_{col}"] = valid_df[col].map(fe)
            test_df[f"feat_frequency_{col}"] = test_df[col].map(fe)
    
    # f15 : 6.133277 -> 6.13113
    if cfg.get('feat_f_2_4', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_2", "f_4")

    # f6 : 6.215775 -> 6.190184
    if cfg.get('feat_f_4_6', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_4", "f_6")

    # f8 : 6.190184 -> 6.168938
    if cfg.get('feat_f_3_15', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_3", "f_15")

    if cfg.get('feat_f_22_31', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_22", "f_31")

    if cfg.get('feat_f_6_15', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_6", "f_15")

    if cfg.get('feat_f_6_18', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_6", "f_18")

    if cfg.get('feat_f_15_18', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_15", "f_18")

    if cfg.get('feat_f_4_15', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_4", "f_15")
    
    # f_22 : 6.13113 -> 6.121204
    if cfg.get('feat_f_13_15', False): 
        train_df, valid_df, test_df = get_group_encoding(train_df, valid_df, test_df, "f_13", "f_15")

    if cfg.get('daily_f_42', False):
        print('daily_f_42') # 직전 전체 기간의 평균 
        all_dates = np.arange(45, 68)
        for date in all_dates:
            if date == 66: # valid
                tmp = train_df.query(f'f_1 <= {date-1}')
                valid_df = get_daily_feat(tmp, valid_df, date, 'f_42', prefix='all_days')
            elif date == 67: # test
                tmp = pd.concat([train_df, valid_df], axis=0).query(f'f_1 <= {date-1}')
                test_df = get_daily_feat(tmp, test_df, date, 'f_42', prefix='all_days')
            else:
                tmp = train_df.query(f'f_1 <= {date-1}')
                train_df = get_daily_feat(tmp, train_df, date, 'f_42', prefix='all_days')
    
    # 만드는 중 
    if cfg.get('hyeon_daily_f_42_v1', False):
        print('hyeon_daily_f_42_v1') # 직전 전체 기간의 평균 
        all_dates = np.arange(45, 68)
        for date in all_dates:
            if date == 66: # valid
                tmp = train_df.loc[(train_df["f_1"] >= date-7) & (train_df["f_1"] < date-1)]
                valid_df = get_daily_feat(tmp, valid_df, date, 'f_42', prefix='7_days')
            elif date == 67: # test
                tmp = pd.concat([train_df, valid_df], axis=0)
                tmp = tmp.loc[(tmp["f_1"] >= date-7) & (tmp["f_1"] < date-1)]
                test_df = get_daily_feat(tmp, test_df, date, 'f_42', prefix='7_days')
            else:
                tmp = train_df.loc[(train_df["f_1"] >= date-7) & (train_df["f_1"] < date-1)]
                train_df = get_daily_feat(tmp, train_df, date, 'f_42', prefix='7_days')

    if cfg.get('hyeon_daily_f_42_v2', False):
        print('hyeon_daily_f_42_v2') # 직전 전체 기간의 평균 
        all_dates = np.arange(45, 68)
        for date in all_dates:
            if date == 66: # valid
                tmp = train_df.loc[(train_df["f_1"] == date-7)]
                valid_df = get_daily_feat(tmp, valid_df, date, 'f_42', prefix='before_7_days')
            elif date == 67: # test
                tmp = pd.concat([train_df, valid_df], axis=0)
                tmp = tmp.loc[(tmp["f_1"] == date-7)]
                test_df = get_daily_feat(tmp, test_df, date, 'f_42', prefix='before_7_days')
            else:
                tmp = train_df.loc[(train_df["f_1"] == date-7)]
                train_df = get_daily_feat(tmp, train_df, date, 'f_42', prefix='before_7_days')

    if cfg.get('hyeon_daily_f_42_v3', False):
        print('hyeon_daily_f_42_v3') # 직전 전체 기간의 평균 
        all_dates = np.arange(45, 68)
        for date in all_dates:
            if date == 66: # valid
                tmp = train_df.loc[(train_df["f_1"] == date-1)]
                valid_df = get_daily_feat(tmp, valid_df, date, 'f_42', prefix='1_days')
            elif date == 67: # test
                tmp = pd.concat([train_df, valid_df], axis=0)
                tmp = tmp.loc[(tmp["f_1"] == date-1)]
                test_df = get_daily_feat(tmp, test_df, date, 'f_42', prefix='1_days')
            else:
                tmp = train_df.loc[(train_df["f_1"] == date-1)]
                train_df = get_daily_feat(tmp, train_df, date, 'f_42', prefix='1_days')

    # fix : 6.074667 -> 6.081853
    if cfg.get('hyeon_click_target_encoding', False):
        print('click_target_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"click_TE_{c}" for c in cat_feats_to_encode]
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            # find some bugs == -> =< 이 맞는데, 수정시 성능 하락함 
            trn_idx = train_df[train_df["f_1"] == t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_clicked"].iloc[trn_idx])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_clicked"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(pd.concat([original_train_cat_df, original_valid_cat_df], axis=0), pd.concat([train_df, valid_df], axis=0)["is_clicked"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    # 굉장히 안좋음. 하루 단위의 cat embedding이 위험성이 큰 듯 
    if cfg.get('hyeon_click_target_encodingv2', False):
        print('click_target_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"click_TEv2_{c}" for c in cat_feats_to_encode]
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            trn_idx = train_df[train_df["f_1"] == t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df.iloc[trn_idx]["is_clicked"])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        trn_idx = train_df[train_df["f_1"] == t+1].index
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df.iloc[trn_idx, :]["is_clicked"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_valid_cat_df, valid_df["is_clicked"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    if cfg.get('hyeon_click_target_encodingv3', False):
        print('click_target_encodingv3')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"click_TEv3_{c}" for c in cat_feats_to_encode]
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            # find some bugs == -> =< 이 맞는데, 수정시 성능 하락함 
            trn_idx = train_df[train_df["f_1"] <= t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_clicked"].iloc[trn_idx])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_clicked"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(pd.concat([original_train_cat_df, original_valid_cat_df], axis=0), pd.concat([train_df, valid_df], axis=0)["is_clicked"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    if cfg.get('hyeon_install_target_encoding', False):
        print('install_target_encoding')
        cat_feats_to_encode = ["f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"install_TE_{c}" for c in cat_feats_to_encode]
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            trn_idx = train_df[train_df["f_1"] == t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_installed"].iloc[trn_idx])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_installed"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(pd.concat([original_train_cat_df, original_valid_cat_df], axis=0), pd.concat([train_df, valid_df], axis=0)["is_installed"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    if cfg.get('click_target_encoding', False):
        print('click_target_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"click_TE_{c}" for c in cat_feats_to_encode]
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        oof = pd.DataFrame([])
        for trn_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(original_train_cat_df, train_df["is_clicked"]):
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_clicked"].iloc[trn_idx])
            oof = pd.concat([oof, ce_target_encoder.transform(original_train_cat_df.iloc[oof_idx, :])], axis=0, ignore_index=False)
        
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_clicked"])
        
        encoded_train_df = oof.sort_index()
        encoded_train_df.columns = encoded_column_names
        train_df = pd.concat([train_df, encoded_train_df], axis=1).reset_index(drop=True)

        encoded_valid_df = ce_target_encoder.transform(original_valid_cat_df)
        encoded_valid_df.columns = encoded_column_names
        encoded_test_df = ce_target_encoder.transform(original_test_cat_df)
        encoded_test_df.columns = encoded_column_names

        valid_df = pd.concat([valid_df, encoded_valid_df], axis=1).reset_index(drop=True)
        test_df = pd.concat([test_df, encoded_test_df], axis=1).reset_index(drop=True)
    
    if cfg.get('install_target_encoding', False):      
        print('install_target_encoding')
        oof = pd.DataFrame([])
        encoded_column_names = [f"install_TE_{c}" for c in cat_feats_to_encode]
        for trn_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(original_train_cat_df, train_df["is_installed"]):
            ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_installed"].iloc[trn_idx])
            oof = pd.concat([oof, ce_target_encoder.transform(original_train_cat_df.iloc[oof_idx, :])], axis=0, ignore_index=False)
        
        ce_target_encoder = ce.TargetEncoder(cols=cat_feats_to_encode, smoothing=cfg.smoothing)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_installed"])
        
        encoded_train_df = oof.sort_index()
        encoded_train_df.columns = encoded_column_names
        train_df = pd.concat([train_df, encoded_train_df], axis=1).reset_index(drop=True)

        encoded_valid_df = ce_target_encoder.transform(original_valid_cat_df)
        encoded_valid_df.columns = encoded_column_names
        encoded_test_df = ce_target_encoder.transform(original_test_cat_df)
        encoded_test_df.columns = encoded_column_names

        valid_df = pd.concat([valid_df, encoded_valid_df], axis=1).reset_index(drop=True)
        test_df = pd.concat([test_df, encoded_test_df], axis=1).reset_index(drop=True)
        gc.collect()

    if cfg.get('hyeon_click_cat_encoding', False):
        print('click_cat_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"click_CE_{c}" for c in cat_feats_to_encode]
        train_df = train_df.sort_values(by="f_1").reset_index(drop=True)
        
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
            trn_idx = train_df[train_df["f_1"] <= t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_clicked"].iloc[trn_idx])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_clicked"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
        ce_target_encoder.fit(pd.concat([original_train_cat_df, original_valid_cat_df], axis=0), pd.concat([train_df, valid_df], axis=0)["is_clicked"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    if cfg.get('hyeon_install_cat_encoding', False):
        print('install_cat_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality
        encoded_column_names = [f"install_CE_{c}" for c in cat_feats_to_encode]
        train_df = train_df.sort_values(by="f_1").reset_index(drop=True)
        
        original_train_cat_df = train_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_valid_cat_df = valid_df[cat_feats_to_encode].reset_index(drop=True).copy()
        original_test_cat_df = test_df[cat_feats_to_encode].reset_index(drop=True).copy()
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        oof = []
        for t in range(45, 65): # 
            ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
            trn_idx = train_df[train_df["f_1"] <= t].index
            val_idx = train_df[train_df["f_1"] == t+1].index
            ce_target_encoder.fit(original_train_cat_df.iloc[trn_idx, :], train_df["is_installed"].iloc[trn_idx])
            temp = ce_target_encoder.transform(original_train_cat_df.iloc[val_idx, :])
            temp["f_0"] = train_df[train_df["f_1"] == t+1]["f_0"]
            oof += [temp]
        
        ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
        ce_target_encoder.fit(original_train_cat_df, train_df["is_installed"])
        temp = ce_target_encoder.transform(original_valid_cat_df)
        temp["f_0"] = valid_df["f_0"]
        oof += [temp]
        oof = pd.concat(oof)
        oof.columns = encoded_column_names + ["f_0"]

        ce_target_encoder = ce.CatBoostEncoder(cols=cat_feats_to_encode)
        ce_target_encoder.fit(pd.concat([original_train_cat_df, original_valid_cat_df], axis=0), pd.concat([train_df, valid_df], axis=0)["is_installed"])
        temp = ce_target_encoder.transform(original_test_cat_df)
        temp["f_0"] = test_df["f_0"]
        temp.columns = encoded_column_names + ["f_0"]

        print(train_df.shape, valid_df.shape, test_df.shape)
        train_df = train_df.merge(oof, how='left', on=["f_0"])
        valid_df = valid_df.merge(oof, how='left', on=["f_0"])
        test_df = test_df.merge(temp, how='left', on=["f_0"])
        print(train_df.shape, valid_df.shape, test_df.shape)

    if cfg.get('feat_cumcountv1', False): 
        print('cumcount_encoding')
        cat_feats_to_encode = ["f_2", "f_4", "f_6", "f_13", "f_15", "f_18"] # high cardinality

        train_df = train_df.sort_values(by=["f_1", "f_0"]).reset_index(drop=True)

        for col in cat_feats_to_encode: 
            train_df[f"cumcount_{col}"] = train_df.groupby([col]).cumcount()
            valid_df[f"cumcount_{col}"] = train_df[f"cumcount_{col}"].max()
            test_df[f"cumcount_{col}"] = train_df[f"cumcount_{col}"].max()
    

    if cfg.get('feat_cumcountv2', False): 
        print('feat_cumcountv2')
        train_df[f"cummean_f42"] = train_df.groupby(["f_2", "f_4"])["f_42"].cumsum() / (1+train_df.groupby(["f_2", "f_4"])["f_42"].cumcount())
        agg = train_df.groupby(["f_2", "f_4"])[["f_2", "f_4", "cummean_f42"]].tail(1)

        valid_df = valid_df.merge(agg, how='left', on=["f_2", "f_4"])
        test_df = test_df.merge(agg, how='left', on=["f_2", "f_4"])

    # f9 : 6.168938 -> 6.133277
    if cfg.get('feat_week', False): 
        # 53~60 : 2, 61~67 : 3
        train_df["week"] = (train_df["f_1"] - 61) // 7 + 3
        valid_df["week"] = (valid_df["f_1"] - 61) // 7 + 3
        test_df["week"] = (test_df["f_1"] - 61) // 7 + 3

    if cfg.get('feat_week_of_days', False): 
        # 53~60 : 2, 61~67 : 3
        train_df["week_of_days"] = train_df["f_1"] % 7
        valid_df["week_of_days"] = valid_df["f_1"] % 7
        test_df["week_of_days"] = test_df["f_1"] % 7

    if cfg.get('feat_new_f_9', False): 
        f_9_dict = {
            0 : {
                869 : 1, 
                21533 : 0
            }, 
            1 : {
                21533 : 1, 
                31372 : 0
            },
            2 : {
                31372 : 1, 
                6675 : 0
            },
            3 : {
                6675 : 1, 
                14659 : 0
            },
            4 : {
                14659 : 1, 
                9638 : 0
            },
            5 : {
                9638 : 1, 
                23218 : 0
            },
            6 : {
                23218 : 1, 
                869 : 0
            }
        }

        train_df["week_of_days"] = train_df["f_1"] % 7
        valid_df["week_of_days"] = valid_df["f_1"] % 7
        test_df["week_of_days"] = test_df["f_1"] % 7

        train_df["new_f_9"] = train_df[["week_of_days", "f_9"]].apply(lambda x: f_9_dict[x[0]][x[1]], axis=1)
        valid_df["new_f_9"] = valid_df[["week_of_days", "f_9"]].apply(lambda x: f_9_dict[x[0]][x[1]], axis=1)
        test_df["new_f_9"] = test_df[["week_of_days", "f_9"]].apply(lambda x: f_9_dict[x[0]][x[1]], axis=1)
        if not cfg.get('feat_week_of_days', False): 
            del train_df["week_of_days"]
            del valid_df["week_of_days"]
            del test_df["week_of_days"]

    if cfg.get('fillna', False):
        print('fillna')
        train_df.fillna(-1, inplace=True)
        valid_df.fillna(-1, inplace=True)
        test_df.fillna(-1, inplace=True)
    
    features = [
        c for c in train_df.columns if c not in [
            'is_clicked', 'is_installed', 'file_name', 'f_0', 'f_1'
            ]  + cfg.delete_features
        ]

    return features, train_df, valid_df, test_df


def get_dataloader(
    cfg,
    train_df=None,
    valid_df=None,
    test_df=None,
    features=None,
    label_col="is_installed"
    ):
    """ Get dataloader
    Args:
        cfg: config files
    Returns:
        dataloader: DataLoader
    """
    if cfg.get('save_data', False):
        train_df.to_parquet(f'{cfg.save_dir}/{cfg.name}/train_df.parquet')
        valid_df.to_parquet(f'{cfg.save_dir}/{cfg.name}/valid_df.parquet')
        test_df.to_parquet(f'{cfg.save_dir}/{cfg.name}/test_df.parquet')

    trn_lgb_data = lgb.Dataset(train_df[features], label=train_df[label_col])
    if cfg.get('sample_weight', False):
        weight_dict = {}
        for i, j in enumerate(range(45, 66)): 
            weight_dict[j] = 0.9 + 0.01 * i
        weight = train_df["f_1"].map(weight_dict).values
        trn_lgb_data = lgb.Dataset(train_df[features], label=train_df[label_col], weight=weight)
    val_lgb_data = lgb.Dataset(valid_df[features], label=valid_df[label_col])
    return (trn_lgb_data, val_lgb_data)

def get_group_encoding(train_df, valid_df, test_df, feat1, feat2): 
    new_feat = feat1 + feat2.replace("f", "")
    train_df[new_feat] = train_df[feat1].astype(str) + '-' + train_df[feat2].astype(str)
    valid_df[new_feat] = valid_df[feat1].astype(str) + '-' + valid_df[feat2].astype(str)
    test_df[new_feat] = test_df[feat1].astype(str) + '-' + test_df[feat2].astype(str) 

    cols = [new_feat] # "f_2", "f_4", "f_6", "f_15", "f_18" 도 적용해보기
    for col in tqdm(cols):
        le = defaultdict(lambda : -1)
        for t in range(45, 65): # 전체 days에 대해서, f_2 - f_33
            for val in train_df[train_df["f_1"] == t][col].unique():
                if val not in le.keys():
                    le[val] = len(le)
            train_df.loc[train_df["f_1"] == t+1, f"{col}_le"] = train_df.loc[train_df["f_1"] == t+1, col].map(le).astype(int)

        for val in train_df[train_df["f_1"] == 65][col].unique():
            if val not in le.keys():
                le[val] = len(le)
        valid_df.loc[:, f"{col}_le"] = valid_df.loc[valid_df["f_1"] == 66, col].map(le).astype(int)
        
        for val in valid_df[col].unique():
            if val not in le.keys():
                le[val] = len(le)
        test_df.loc[:, f"{col}_le"] = test_df.loc[:, col].map(le).astype(int)

    for col in [f"{col}_le" for col in cols]: 
        train_df[col] = train_df[col].astype(float)
        valid_df[col] = valid_df[col].astype(float)            
        test_df[col] = test_df[col].astype(float)

    for col in cols: 
        le = defaultdict(lambda : -1)
        for val in train_df[col].unique():
            if val not in le.keys():
                le[val] = len(le)
        train_df[col] = train_df[col].map(le).astype(float)
        valid_df[col] = valid_df[col].map(le).astype(float)

        for val in valid_df[col].unique():
            if val not in le.keys():
                le[val] = len(le)
        test_df[col] = test_df[col].map(le).astype(float)
    return train_df, valid_df, test_df     

def get_daily_feat(df, ret_df, date, feat, prefix):
    _feat_val = df[f'{feat}'].mean()
    ret_df.loc[ret_df['f_1'] == date, f'daily_{feat}_{prefix}'] = _feat_val
    return ret_df