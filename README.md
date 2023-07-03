# challenges-recsys-challenge-2023

## Dataset Schema 
```
├── configs
│   ├── default_config.py
│   └── submit.py
├── data
│   └── common.py
├── models
│   └── common.py
├── README.md
├── preprocess.py
├── requirements.txt
└── run.py
```

## Preprocess
```python preprocess.py --config submit```

Format all csv files to one parquet file.
- Input
    - train 000000000000.csv ~ 000000000030.csv
    - test 000000000000.csv
- Output
    - train.paruqet
    - test.parquet

## Train & Inference 
```python run.py --config submit --no_wandb --verbose```

- Input 
    - train.parquet
    - test.parquet 
- Output 
    - submission file
    - config.py 
    - feature_importance.png 

## Solution Summary 
- Preprocessing
    - count preprocessing : Divide into the second smallest value.
    - remove features : Using Adversarial Validation (Remove Train-Test gap features)

- Feature Engineering 
    - featv1 : Encodes the group category for the current month using the number of value counts from the previous month.
        - feat_f_2_4, feat_f_4_6, feat_f_3_15, feat_f_13_15
    - featv2 : Day of week feature
        - feat_week  
    - featv3 : f_42 / f_42 of previous period  
        - daily_f_42 : previous all days 
        - hyeon_daily_f_42_v1 : previous 7 days 
    - featv4 : Catboost encoding of categorical features using previous all days 
        - hyeon_click_cat_encoding : using target click
        - hyeon_install_cat_encoding : using target install
    - feat_f_56 : f_51 / (f_56 + 1)

- Validation 
    - Train : f_1 < 66 
    - Valid : f_1 = 66

- Modeling 
    - LightGBM with categorical features 