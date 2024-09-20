# src/utils/data_utils.py

import pandas as pd
from pyxlsb import open_workbook as open_xlsb

def read_xlsb(file):
    data = []
    with open_xlsb(file) as wb:
        with wb.get_sheet(1) as sheet:
            for row in sheet.rows():
                data.append([item.v for item in row])

    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def load_and_preprocess_data(file_path):
    df = read_xlsb(file_path)
    df = df.dropna(how='all', axis=1)
    df['Opdate'] = pd.to_datetime(df['Opdate'], unit='D', origin='1898-12-30')
    return df