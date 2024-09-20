# src/clinical_table_analysis.py

import pandas as pd
import numpy as np
from utils.clinical_utils import characteristics_table, compute_ASMD, calculate_counts_and_proportions
from utils.preprocessing_utils import N_categorize

def prepare_data(df):
    df['Primary_Site'].fillna('mid', inplace=True)
    df['T_category'] = df['pTNM7_1'].str.extract('(TX|T0|Tis|T1a|T1b|T2|T3|T4a|T4b)').replace('T', '', regex=True)
    df['N_category'] = df.total_pos_LN.apply(N_categorize)
    df['M_category'] = df['pTNM7_1'].str.extract('(M0|M1)').replace('M', '', regex=True)
    
    T1_group = df[df['T_category'].isin(['1a', '1b'])]
    T2_4_Group = df[df['T_category'].isin(['2', '3', '4a', '4b'])]
    
    return T1_group, T2_4_Group

def create_clinical_table(T1_group, T2_4_Group):
    T1_group_data = characteristics_table(T1_group[T1_group.N_category!='0'])
    T2_4_group_data = characteristics_table(T2_4_Group[T2_4_Group.N_category!='0'])
    
    ASMD_result = compute_ASMD(T1_group, T1_group_data, T2_4_Group, T2_4_group_data)
    
    column_names = {
        'T_1': f"pT1 (N = {len(T1_group[T1_group.N_category!='0'])})",
        'T_2_4': f"pT2-4 (N = {len(T2_4_Group[T2_4_Group.N_category!='0'])})"
    }
    
    clinical_table_df = pd.DataFrame({
        column_names['T_1']: T1_group_data,
        column_names['T_2_4']: T2_4_group_data,
        "ASMD": ASMD_result
    })
    
    return clinical_table_df

def create_frequency_table(df):
    T1_df = df[df['T_category'].isin(['1a', '1b'])]
    T24_df = df[df['T_category'].isin(['2', '3', '4a', '4b'])]

    subgroup_dfs = {
        'T1_upper': T1_df[T1_df.Primary_Site=='upper'][T1_df.N_category!='0'],
        'T24_upper': T24_df[T24_df.Primary_Site=='upper'][T24_df.N_category!='0'],
        'T1_mid': T1_df[T1_df.Primary_Site=='mid'][T1_df.N_category!='0'],
        'T24_mid': T24_df[T24_df.Primary_Site=='mid'][T24_df.N_category!='0'],
        'T1_lower': T1_df[T1_df.Primary_Site=='lower'][T1_df.N_category!='0'],
        'T24_lower': T24_df[T24_df.Primary_Site=='lower'][T24_df.N_category!='0']
    }

    pos_columns = [col for col in df.columns if col.startswith("pos_")]
    results_dict = {}

    for group_name, subgroup_df in subgroup_dfs.items():
        header_with_count = f"{group_name} (N={subgroup_df.shape[0]})"
        counts, proportions = calculate_counts_and_proportions(subgroup_df, pos_columns)
        combined_results = [f"{c} ({p}%)" for c, p in zip(counts, proportions)]
        results_dict[header_with_count] = combined_results

    rows_to_remove = [
        'pos_ON', 'pos_neckLN', 'pos_OM', 'pos_mediaLN', 'pos_OA',
        'pos_abdoLN', 'pos_neckLN_new', 'pos_mediaLN_new', 'pos_abdoLN_new'
    ]

    results_df = pd.DataFrame(results_dict, index=pos_columns)
    results_df = results_df.drop(rows_to_remove, errors='ignore')

    return results_df

def main():
    df = pd.read_csv("../data/preprocessed/ECA_Dataset.csv")
    T1_group, T2_4_Group = prepare_data(df)
    
    clinical_table_df = create_clinical_table(T1_group, T2_4_Group)
    clinical_table_df.to_csv("../data/preprocessed/clinical_table.csv")
    
    frequency_table_df = create_frequency_table(df)
    frequency_table_df.to_csv("../data/preprocessed/frequency_table.csv")
    
    print("Clinical table and frequency table have been created and saved.")
    
    return clinical_table_df, frequency_table_df

if __name__ == "__main__":
    main()