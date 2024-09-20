# src/utils/preprocessing_utils.py

import pandas as pd
import numpy as np

def merge_and_rename_columns(df, mapping_dict):
    df_copy = df.copy()
    
    for original_col in mapping_dict.keys():
        df_copy[original_col] = pd.to_numeric(df_copy[original_col], errors='coerce').fillna(0)
        
        if mapping_dict[original_col] not in df_copy:
            df_copy[mapping_dict[original_col]] = 0
        
        df_copy[mapping_dict[original_col]] += df_copy[original_col]
        df_copy.drop(original_col, axis=1, inplace=True)
    
    target_cols = list(mapping_dict.values())
    for target_col in target_cols:
        df_copy[target_col].fillna(0, inplace=True)
    
    return df_copy

def verify_merging(original_df, new_df, mapping_dict):
    discrepancies = []

    for old_column, new_column in mapping_dict.items():
        if old_column in original_df.columns:
            if new_column not in new_df.columns:
                discrepancies.append(f"New column {new_column} missing in the merged dataframe.")
                continue
            expected_sum = original_df[old_column].sum()
            actual_sum = new_df[new_column].sum()
            
            if not np.isclose(expected_sum, actual_sum):
                discrepancies.append(f"Discrepancy in {new_column}: Expected {expected_sum}, but got {actual_sum}.")

    return discrepancies

def calculate_sums(df):
    df_copy = df.copy()
    
    df_copy['pos_neckLN_new'] = df_copy[['pos_101R', 'pos_101L', 'pos_102R', 'pos_102L', 'pos_104R', 'pos_104L']].sum(axis=1)
    df_copy['pos_mediaLN_new'] = df_copy[['pos_106preR', 'pos_106preL', 'pos_106recR', 'pos_106recL', 'pos_107', 'pos_105/108/110', 'pos_112pulR', 'pos_112pulL']].sum(axis=1)
    df_copy['pos_abdoLN_new'] = df_copy[['pos_1/2/7', 'pos_8', 'pos_9']].sum(axis=1)
    df_copy['total_pos_LN_new'] = df_copy[['pos_mediaLN_new', 'pos_abdoLN_new', 'pos_neckLN_new']].sum(axis=1)
    
    return df_copy

def N_categorize(x):
    if x == 0:
        return '0'
    elif 1 <= x <= 2:
        return '1'
    elif 3 <= x <= 6:
        return '2'
    else:
        return '3'