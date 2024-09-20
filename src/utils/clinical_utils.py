# src/utils/clinical_utils.py

import pandas as pd
import numpy as np

def characteristics_table(group):
    """Extract and format baseline characteristics for a given group."""
    combined_count = group['T_category'].value_counts().get('1a', 0) + group['T_category'].value_counts().get('Tis', 0)
    combined_percentage = combined_count / len(group) * 100
    
    data = {
        'Age': f"{group['Age'].mean():.2f} ({group['Age'].std():.2f})",
        'Male Sex': f"{group['Sex'].value_counts().get('M', 0):,.0f} ({group['Sex'].value_counts(normalize=True).get('M', 0) * 100:.1f}%)",
        'Total Dissected LN': f"{group['total_LN'].mean():.2f} ({group['total_LN'].std():.2f})",
        'Primary Site Upper': f"{group['Primary_Site'].value_counts().get('upper', 0):,.0f} ({group['Primary_Site'].value_counts(normalize=True).get('upper', 0) * 100:.1f}%)",
        'Primary Site Mid': f"{group['Primary_Site'].value_counts().get('mid', 0):,.0f} ({group['Primary_Site'].value_counts(normalize=True).get('mid', 0) * 100:.1f}%)",
        'Primary Site Lower': f"{group['Primary_Site'].value_counts().get('lower', 0):,.0f} ({group['Primary_Site'].value_counts(normalize=True).get('lower', 0) * 100:.1f}%)",
        'pTis&pT1a': f"{combined_count:,.0f} ({combined_percentage:.1f}%)",
        'pT1b': f"{group['T_category'].value_counts().get('1b', 0):,.0f} ({group['T_category'].value_counts(normalize=True).get('1b', 0) * 100:.1f}%)",
        'pT2': f"{group['T_category'].value_counts().get('2', 0):,.0f} ({group['T_category'].value_counts(normalize=True).get('2', 0) * 100:.1f}%)",
        'pT3': f"{group['T_category'].value_counts().get('3', 0):,.0f} ({group['T_category'].value_counts(normalize=True).get('3', 0) * 100:.1f}%)",
        'pT4a': f"{group['T_category'].value_counts().get('4a', 0):,.0f} ({group['T_category'].value_counts(normalize=True).get('4a', 0) * 100:.1f}%)",
        'pT4b': f"{group['T_category'].value_counts().get('4b', 0):,.0f} ({group['T_category'].value_counts(normalize=True).get('4b', 0) * 100:.1f}%)",
        'pNx': f"{group['N_category'].value_counts().get('x', 0):,.0f} ({group['N_category'].value_counts(normalize=True).get('x', 0) * 100:.1f}%)",
        'pN0': f"{group['N_category'].value_counts().get('0', 0):,.0f} ({group['N_category'].value_counts(normalize=True).get('0', 0) * 100:.1f}%)",
        'pN1': f"{group['N_category'].value_counts().get('1', 0):,.0f} ({group['N_category'].value_counts(normalize=True).get('1', 0) * 100:.1f}%)",
        'pN2': f"{group['N_category'].value_counts().get('2', 0):,.0f} ({group['N_category'].value_counts(normalize=True).get('2', 0) * 100:.1f}%)",
        'pN3': f"{group['N_category'].value_counts().get('3', 0):,.0f} ({group['N_category'].value_counts(normalize=True).get('3', 0) * 100:.1f}%)",
    }
    return data

def compute_cohens_h(P_Counts, P_total, N_Counts, N_total):
    """Compute Cohen's H given counts and total numbers."""
    P_proportion = P_Counts / P_total
    N_proportion = N_Counts / N_total
    
    h = 2 * np.arcsin(np.sqrt(P_proportion)) - 2 * np.arcsin(np.sqrt(N_proportion))
    return round(abs(h), 3)

def compute_cohens_d(P_mean, P_std, P_total, N_mean, N_std, N_total):
    """Compute Cohen's D for continuous data given means, standard deviations, and total numbers."""
    pooled_std = np.sqrt(((N_total-1)*(N_std**2) + (P_total-1)*(P_std**2))/(N_total+P_total-2))
    d = (P_mean - N_mean) / pooled_std
    return round(abs(d), 3)

def compute_ASMD(group_1, group_1_data, group_2, group_2_data):
    ASMD = {}
    for key in group_1_data.keys():
        if key == "Age" or key == "Total Dissected LN":
            P_mean = float(group_1_data[key].split()[0])
            N_mean = float(group_2_data[key].split()[0])
            P_std = float(group_1_data[key].split()[1].strip('()'))
            N_std = float(group_2_data[key].split()[1].strip('()'))
            ASMD[key] = compute_cohens_d(P_mean, P_std, len(group_1), N_mean, N_std, len(group_2))
        else:
            P_value = int(group_1_data[key].split()[0].replace(',', ''))
            N_value = int(group_2_data[key].split()[0].replace(',', ''))
            ASMD[key] = compute_cohens_h(P_value, len(group_1), N_value, len(group_2))
    return ASMD

def calculate_counts_and_proportions(df, pos_columns):
    df_clipped = df[pos_columns].clip(upper=1)
    counts = df_clipped.sum().astype(int)
    proportions = (counts / df_clipped.shape[0] * 100).round(2)
    return counts, proportions