# src/data_preprocessing.py

import pandas as pd
import numpy as np
from utils.data_utils import load_and_preprocess_data
from utils.preprocessing_utils import merge_and_rename_columns, calculate_sums, N_categorize

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("../data/original/1994-2022_ECA_Data.xlsb")

    # Define columns to remove and mapping dictionaries
    columns_to_remove = [
        "misLN_text\n(2019년도부터는 재코딩필요, neck, abd. Field없어서 other로 들어가 있는 것들이 많음)",
        "pos_misLN (G)",
        "total_misLN (H)",
        "pathsite_1",
        "Diff1_coding",
        "RM_1_text",
        "RM_1",
        "pStage7_1", "pT7 ", 
        "pN7 ", 
        "pM7 ", 
        "pStage7 "
    ]

    df = df.drop(columns=columns_to_remove)

    # Define renaming dictionary and column order
    renaming_dict = {
        "total_pos_LN_수식(A+C+E+G)": "total_pos_LN",
        "total_LN_수식(B+D+F+H)": "total_LN",
        "n_No": "Patient_Number",
        "Opdate": "Operation_Date",
        "lesion1": "lesion_text",
        "pos_101R\n(RC1)" : "pos_101R",
        "pos_102R\n(RC2)" : "pos_102R",
        "pos_101L \n(LC1)" : "pos_101L",
        "pos_102L\n(LC2)" : "pos_102L",
        "pos_103R\n(RC3)" : "pos_103R",
        "pos_103L\n(LC3)" : "pos_103L",
        "pos_104R\n(SCN, Rt.)" : "pos_104R",
        "pos_104L\n(SCN, Lt.)" : "pos_104L",
        "pos_level3R \n(Rt. level 3)" : "pos_level3R",
        "pos_level3L \n(Lt. level 3)" : "pos_level3L", 
        "pos_level4R \n(Rt. level 4)" : "pos_level4R", 
        "pos_level4L \n(Lt. level 4)" : "pos_level4L", 
        "pos_level5R \n(Rt. level 5)" : "pos_level5R",
        "pos_level5L \n(Lt. level 5)" : "pos_level5L",
        "pos_level6R \n(Rt. level 6)" : "pos_level6R",
        "pos_level6L \n(Lt. level 6)" : "pos_level6L",
        "pos_neckLN (A)": "pos_neckLN",
        "total_neckLN (B)": "total_neckLN",
        "pos_1R\n(=low cervical, supraclavisular, sternal notch nodes)\n(19년도부터는 neck site LN field abd. LN field없음)" : "pos_1R",
        "total_1R\n(=low cervical, supraclavisular, sternal notch nodes)" : "total_1R",
        "pos_1L\n(=low cervical, supraclavisular, sternal notch nodes)" : "pos_1L",
        "total_1L\n(=low cervical, supraclavisular, sternal notch nodes)" : "total_1L",
        "pos_3(2019~)": "pos_3",
        "total_3(2019~)": "total_3",
        "pos_2R_4R\n(2019~)": "pos_2R_4R",
        "total_2R_4R\n(2019~)": "total_2R_4R", 
        "pos_2L_4L\n(2019~)": "pos_2L_4L",
        "total_2L_4L\n(2019~)": "total_2L_4L",
        "pos_5(subaortic LN)": "pos_5",
        "pos_6\n(2019~)": "pos_6",
        "pos_7(subcarina LN)": "pos_7",
        "pos_8(periesophageal LN(8U, 8M, 8L)": "pos_8",
        "pos_9R(pulmonary ligament LN)": "pos_9R",
        "pos_9L(pulmonary ligament LN)": "pos_9L",
        "pos_10R(tracheobronchial LN)": "pos_10R",
        "pos_10L(tracheobronchial LN)": "pos_10L",
        "pos_11R\n(2019~)": "pos_11R",
        "total_11R\n(2019~)": "total_11R",
        "pos_11L\n(2019~)": "pos_11L",
        "total_11L\n(2019~)": "total_11L",
        "pos_12R\n(2019~)": "pos_12R",
        "total_12R\n(2019~)": "total_12R",
        "pos_12L\n(2019~)": "pos_12L",
        "total_12L\n(2019~)": "total_12L",
        "pos_13R\n(2019~)": "pos_13R",
        "total_13R\n(2019~)": "total_13R",
        "pos_13L\n(2019~)": "pos_13L",
        "total_13L\n(2019~)": "total_13L",
        "pos_14R\n(2019~)": "pos_14R",
        "total_14R\n(2019~)": "total_14R",
        "pos_14L\n(2019~)": "pos_14L",
        "total_14L\n(2019~)": "total_14L",
        "total_6\n(2019~)": "total_6",
        "pos_15R\n(periesophageal diaphragmatic LN, RD)\n": "pos_15R",
        "pos_15L\n(periesophageal diaphragmatic LN, LD)": "pos_15L",
        "pos_regional\n(2019~)": "pos_regional",
        "total_regional\n(2019~)": "total_regional",
        "pos_mediaLN (C)": "pos_mediaLN",
        "total_mediaLN (D)": "total_mediaLN",
        "pos_G3\n(#16=Paracardial LN(perigastric LN), #17=Left gastirc LN, G4)": "pos_G3",
        "total_G3\n(#16=Paracardial LN(perigastric LN), #17=Left gastirc LN, G4)": "total_G3",
        "pos_G1\n(#18=common hepatic LN)": "pos_G1",
        "total_G1\n(#18=common hepatic LN)": "total_G1",
        "pos_G2(#19=splenic LN, #20=celiac LN)": "pos_G2",
        "total_G2(#19=splenic LN, #20=celiac LN)": "total_G2",
        "pos_abdoLN (E)": "pos_abdoLN",
        "total_abdoLN (F)": "total_abdoLN",
        "pos_other_neck": "pos_ON",
        "total_other_neck": "total_ON",
        "pos_other_media": "pos_OM",
        "total_other_media": "total_OM",
        "pos_other_abdo": "pos_OA",
        "total_other_abdo": "total_OA",
        "pos_regional": "pos_REG",
        "total_regional": "total_REG"
    }

    column_order = ['Patient_Number', 'Sex', 'Age','Year','Operation_Date','IC_cm','lesion_text','Lesion_Coding',
    'pos_101R','total_101R','pos_101L','total_101L','pos_102R','total_102R','pos_102L','total_102L',
    'pos_104R', 'total_104R', 'pos_104L', 'total_104L', 'other_text_neck', 
    'pos_ON', 'total_ON', 'pos_neckLN', 'total_neckLN',
    'pos_106preR', 'total_106preR', 'pos_106preL', 'total_106preL', 'pos_106recR', 'total_106recR', 'pos_106recL', 'total_106recL', 
    'pos_107', 'total_107', 'pos_105/108/110', 'total_105/108/110', 'pos_112pulR', 'total_112pulR', 'pos_112pulL', 'total_112pulL',
    'other_text_media', 'pos_OM', 'total_OM', 'pos_mediaLN', 'total_mediaLN',
    'pos_1/2/7', 'total_1/2/7', 'pos_8', 'total_8', 'pos_9', 'total_9', 'other_text_abdo', 'pos_OA', 'total_OA',
    'pos_abdoLN', 'total_abdoLN', 'total_pos_LN', 'total_LN', 'pTNM7_1']

    # Apply renaming and reordering
    df = df.rename(columns=renaming_dict)
    df = df[column_order]

    # Apply merging and calculate sums
    mapping_dict = {
        "pos_103R": "pos_104R_new",
        "pos_104R": "pos_104R_new",
        "pos_103L": "pos_104L_new",
        "pos_104L": "pos_104L_new",
        "pos_level3L": "pos_ON_new",
        "pos_level3R": "pos_ON_new",
        "pos_level4L": "pos_ON_new",
        "pos_level4R": "pos_ON_new",
        "pos_level5L": "pos_ON_new",
        "pos_level5R": "pos_ON_new",
        "pos_level6L": "pos_ON_new",
        "pos_level6R": "pos_ON_new",
        "pos_2R": "pos_106preR_new",
        "pos_2L": "pos_106preL_new",
        "pos_4R": "pos_106preR_new",
        "pos_4L": "pos_106preL_new",
        "pos_2R_4R": "pos_106preR_new",
        "pos_2L_4L": "pos_106preL_new",
        "pos_RRLN": "pos_106recR_new",
        "pos_LRLN": "pos_106recL_new",
        "pos_1R": "pos_106recR_new",
        "pos_1L": "pos_106recL_new",
        "pos_7": "pos_107_new",
        "pos_8": "pos_105/108/110_new",
        "pos_10R": "pos_105/108/110_new",
        "pos_10L": "pos_105/108/110_new",
        "pos_9R": "pos_112pulR_new",
        "pos_9L": "pos_112pulL_new",
        "pos_3": "pos_OM_new",
        "pos_5": "pos_OM_new",
        "pos_6": "pos_OM_new",
        "pos_11R": "pos_OM_new",
        "pos_11L": "pos_OM_new",
        "pos_12R": "pos_OM_new",
        "pos_12L": "pos_OM_new",
        "pos_13R": "pos_OM_new",
        "pos_13L": "pos_OM_new",
        "pos_14R": "pos_OM_new",
        "pos_14L": "pos_OM_new",
        "pos_15R": "pos_OM_new",
        "pos_15L": "pos_OM_new",
        "pos_regional": "pos_OM_new",
        "pos_G3": "pos_1/2/7_new",
        "pos_G1": "pos_8_new",
        "pos_G2": "pos_9_new"
    }

    merged_df = merge_and_rename_columns(df, mapping_dict)
    ordered_df = calculate_sums(merged_df)

    # Final processing steps
    exclude_columns = [
        "Patient_Number", "Sex", "Age", "Year", "Operation_Date", "IC_cm",
        "lesion_text", "Lesion_Coding", "other_text_neck", "other_text_media",
        "other_text_abdo", "pTNM7_1", "pStage7_1", "pT7", "pN7", "pM7", "pStage7"
    ]

    numerical_columns = [col for col in ordered_df.columns if col not in exclude_columns and ordered_df[col].dtype in ['float64', 'int64']]

    for col in numerical_columns:
        ordered_df[col] = ordered_df[col].apply(lambda x: round(x) if pd.notnull(x) else x)

    primary_sites_mapping = {1: 'upper', 2: 'mid', 3: 'lower'}
    ordered_df['Primary_Site'] = ordered_df['Lesion_Coding'].map(primary_sites_mapping)

    # Save the processed data
    ordered_df.to_csv("../data/preprocessed/ECA_Dataset.csv", index=False)

    print("Data preprocessing completed. Processed data saved to '../data/preprocessed/ECA_Dataset.csv'")

if __name__ == "__main__":
    main()