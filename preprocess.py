import pandas as pd 
import numpy as np 
import os, glob
import argparse
import importlib
from tqdm import tqdm 

def main():
    # parse arguments
    args = parse_arguments()
    cfg = importlib.import_module(f'configs.{args.config}').cfg
    cfg.update(args)

    tr_path = os.path.join(cfg.root_path, "train")
    total_data = []
    for file_name in tqdm(sorted(glob.glob(tr_path + "/*"))):
        print(file_name)
        data = pd.read_csv(file_name, sep='\t')
        data["file_name"] = file_name.split('/')[-1].split('.')[0]
        total_data += [data]

    total_data = pd.concat(total_data)
    os.makedirs(tr_path, exist_ok=True)
    total_data.to_parquet(os.path.join(tr_path, "train.parquet"), index=False)

    te_path = os.path.join(cfg.root_path, "test")
    os.makedirs(te_path,  exist_ok=True)
    test_data = pd.read_csv(os.path.join(te_path, "000000000000.csv"), sep='\t')
    test_data.to_parquet(os.path.join(te_path, "test.parquet"), index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="recsys challenge 2023")
    parser.add_argument(
        '--config', required=True,
        help="config filename")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()