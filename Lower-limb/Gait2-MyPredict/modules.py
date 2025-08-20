import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt

import sys
sys.path.append('../Shared')
import processing, Model, Visualization


def label_extraction(data):
    # --- (1) Label 추출 ---
    y = data["Label"][:].astype(float).ravel()
    valid_idx = np.where((y != -2) & (y != -1))[0] # 불필요한 레이블(-2, -1) 제거
    y = y[valid_idx]
    return y, valid_idx

def stack_signals(trial, keys, valid_idx):
    # --- (2) Modality 별 데이터 추출 ---
    """여러 channel(dataset)을 axis=1 로 합치기"""
    data_list = []
    for k in keys:
        d = trial[k][:]
        if d.ndim == 1:
            d = d[:, np.newaxis]  # (samples,) -> (samples, 1)
        data_list.append(d)
    data = np.hstack(data_list)   # (samples, channels)
    return data[valid_idx]


def Downsample_to_balance_class(y, ):
    # --- (3) Downsampling to balance classes ---
    df = pd.DataFrame({"y": y})

    balanced_idx = []
    for lbl in np.unique(y):
        lbl_idx = np.where(y == lbl)[0]
        min_n = df["y"].value_counts().min()  # 가장 작은 클래스 크기
        sampled_idx = resample(lbl_idx, n_samples=min_n, random_state=42, replace=False)
        balanced_idx.extend(sampled_idx)

    balanced_idx = np.array(balanced_idx)
    return balanced_idx

def random_downsample(X, y=None, fraction=0.5, random_state=42):
    # --- (4) when the data is too much, we randomly remove samples by fraction % ---
    #fraction: 남길 비율

    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_keep = int(n_samples * fraction)
    idx_keep = np.random.choice(n_samples, size=n_keep, replace=False)

    X_down = X[idx_keep]

    if y is not None:
        y_down = y[idx_keep]
        return X_down, y_down
    else:
        return X_down

def get_X_y(data_list, label_list):
    all_X, all_y = [], []

    for data, label in zip(data_list, label_list):
        feat = processing.extract_features(data)  # (num_channels*5,)
        all_X.append(feat)
        all_y.append(label)

    all_X, all_y = np.array(all_X), np.array(all_y)   # (N, num_channels*5)

    return all_X, all_y


def y_change_to_int(y_lst):
    unique_labels = np.unique(y_lst)
    print("Unique labels:", unique_labels)

    # 매핑: 원래 레이블 -> 0부터 시작하는 정수
    label_map = {orig: i for i, orig in enumerate(unique_labels)}
    y_int = np.array([label_map[v] for v in y_lst])

    unique_labels = np.unique(y_int)
    print("New labels:", unique_labels)

    return y_int
