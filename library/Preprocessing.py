# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:02:30 2025

@author: ht
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from scipy.linalg import null_space
from scipy.optimize import minimize
import itertools
import os

base_folder = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_dataframe_targets(
    df, 
    n_samples=500,
    n_features=6,
    n_targets=5
):
     
    # Step 1: Define linear constraints A y = b
    def generate_list(n):
        result = []
        for i in range(n_targets//2,n,2):
            row = [0] * n
            row[i-1] = 1
            row[i] = -1
            result.append(row)
        return result
    temp = generate_list(n_targets)
    temp2 = [[1 if i <n_targets//2 else 0 for i in range(n_targets)]]
    
    A = np.array(temp2+temp)
    
    tempb = [1] + [(i+1)/10 for i in range(len(temp))]
    b = np.array(tempb)
    target_cols = [f'y{id + 1}' for id in range(n_targets)]
    # feature_cols = [f'X{id + 1}' for id in range(n_features)] 

    # features_df = df[feature_cols]
    # targets_df = df[target_cols]

    # Step 2: Project Z onto the affine constraint manifold Ay = b    
    def project_to_constraints(z):
        # 1. Compute analytical projection to linear manifold (ignoring bounds first for speed)
        A_pinv = A @ A.T
        lambda_vec = np.linalg.solve(A_pinv, A @ z - b)
        y_linear = z - A.T @ lambda_vec
        
        # If analytical solution satisfies non-negativity, return it
        if np.all(y_linear >= 0):
            return y_linear

        # 2. If violated, solve QP: min ||y - z||^2 s.t. Ay = b, y >= 0
        def objective(y):
            return np.sum((y - z)**2)

        constraints = [
            {'type': 'eq', 'fun': lambda y: A @ y - b}
        ]
        # Non-negative bounds
        bounds = [(0, None) for _ in range(len(z))]

        # Use y_linear as initial guess to speed up convergence
        result = minimize(objective, y_linear, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    Y = df[target_cols].to_numpy(dtype=float)
    Y_proj = np.array([project_to_constraints(y) for y in Y])
    Y_proj[np.abs(Y_proj) < 1e-6] = 0.0
    
    df_out = df.copy()
    df_out.loc[:, target_cols] = Y_proj

    return df_out
