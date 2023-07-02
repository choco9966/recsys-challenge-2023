from configs.default_config import cfg
import os

# common
cfg.api_key = '1e2f64d1af3597c5a910968d7eeaa8ec360444ad'
cfg.project = 'recsys-challenge-2023.lightgbm'
cfg.user_name = 'Hyeonwoo'
cfg.device = 'cuda' # 2
cfg.name = f'{__name__}'

# dataset
cfg.filtering = False 
cfg.sub_training = False # 최근 데이터만 사용 (f_1 is in [55~66]) # 55 이전의 기저 CVR과 이후의 CVR이 꽤 차이가 남 # 그래도 LB 상으론 다 쓰는게 좋음
cfg.remove_first_day = False
cfg.split = "time" # time, gkfold, skfold, tscv
cfg.save_data = False 

# model
cfg.num_iterations = 10000

# feature 
cfg.cyclical_feats = False # f_1 -> day_of_week로 변환 후 재시도 (성능향상: 6.20401 -> 6.193062)
cfg.filter_features_by_importance = False
cfg.fillna = False
cfg.feat_log_continuous = False
cfg.feat_frequency_encoding = False
cfg.feat_frequency_encoding_7days = True
cfg.feat_frequency_encoding_full_daysv3 = True 

cfg.feat_count_preprocessing = True

cfg.feat_f_42 = False
cfg.feat_f_43 = False
cfg.feat_f_56 = True 

cfg.feat_f_2_4 = True 
cfg.feat_f_4_6 = True 
cfg.feat_f_3_15 = True
cfg.feat_f_6_15 = False
cfg.feat_f_6_18 = False
cfg.feat_f_15_18 = False
cfg.feat_f_4_15 = False
cfg.feat_f_13_15 = True
cfg.feat_week = True
cfg.feat_week_of_days = False
cfg.feat_f_others = False

cfg.feat_f_4_6_f_42 = False 
cfg.feat_f_3_15_f_42 = False
cfg.feat_new_f_9 = False
cfg.daily_f_42 = True
cfg.hyeon_daily_f_42_v1 = True
cfg.install_target_encoding = False
cfg.click_target_encoding = False
cfg.hyeon_click_target_encoding = False
cfg.hyeon_click_cat_encoding = True
cfg.hyeon_install_cat_encoding = True

# cfg.delete_features = ["f_2", "f_4", "f_6"] # f_19, f_20, f_21, f_56, f_57, f_61, f_76
cfg.categorical_features = ["f_2_count_full_7days", "f_3", "f_4_count_full_7days", "f_6_count_full_7days", "f_13_count_full_7days", "f_15_count_full_7days", "f_18_count_full_7days"]
cfg.delete_features = ['f_4', 'f_12', 'f_7', 'f_7_count_full', 'f_9', 'f_11', 'f_43', 'f_51', 'f_58', 'f_59', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70']

# etc
cfg.save_dir = 'results'
cfg.save_file = 'submission.csv'
cfg.root_path = "/data/project/recsys-challenge-2023/sharechat_recsys2023_data/"
cfg.train_file = os.path.join(cfg.root_path, "train/train.parquet")
cfg.test_file = os.path.join(cfg.root_path, "test/test.parquet")
cfg.label_col = "is_installed"
cfg.stacking = False
cfg.downcast = True
cfg.use_train_valid = True # True면 성능 하락

cfg.params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'binary_logloss',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }

cfg.params_full = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }