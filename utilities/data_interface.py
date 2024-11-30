# -*- coding: utf-8 -*-
import pandas as pd

def read_dataset(file_path):
    df = pd.read_csv(
        file_path
    )
    df.rename(
        columns={
            'PAY_0':'PAY_1',
            'default.payment.next.month':'def_pay'
        },
        inplace=True
    )
    df.drop(['ID'], axis=1, inplace=True)
    return df


def get_data(file_path):
    data = read_dataset(file_path)
    features = [
       'LIMIT_BAL',
       'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    ]
    y = data.def_pay.copy()
    X = data[features].copy()
    return X,y
