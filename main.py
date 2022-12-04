import pickle
import re
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pandas import Index
from pydantic import BaseModel

path_to_model = '/home/user/Desktop/indianCarCost/pickle_model.pkl'
path_to_scaler = '/home/user/Desktop/indianCarCost/pickle_scaler.pkl'
path_to_mean = '/home/user/Desktop/indianCarCost/mean.pkl'

app = FastAPI()

x_cols = Index(['year', 'km_driven', 'mileage', 'engine', 'max_power', 'fuel_Diesel',
                'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
                'seller_type_Trustmark Dealer', 'transmission_Manual',
                'owner_Fourth & Above Owner', 'owner_Second Owner',
                'owner_Test Drive Car', 'owner_Third Owner', 'seats_14', 'seats_2',
                'seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9'],
               dtype='object')


class Item(BaseModel):
    name: Optional[str] = np.nan
    year: Optional[int] = np.nan
    selling_price: Optional[int] = np.nan
    km_driven: Optional[int] = np.nan
    fuel: Optional[str] = np.nan
    seller_type: Optional[str] = np.nan
    transmission: Optional[str] = np.nan
    owner: Optional[str] = np.nan
    mileage: Optional[str] = np.nan
    engine: Optional[str] = np.nan
    max_power: Optional[str] = np.nan
    torque: Optional[str] = np.nan
    seats: Optional[float] = np.nan

    def return_df(self):
        return pd.DataFrame(self.__dict__, index=[0])


class Items(BaseModel):
    objects: List[Item]


def prepare_df(df):
    df = df.drop(["selling_price", "name", 'torque'], axis=1)
    for i, row in df.iterrows():
        if re.search('\d+', df.at[i, 'max_power']) is None:
            # df.at[i, 'max_power'] == ' bhp' or df.at[i, 'max_power'] == np.nan or df.at[i, 'max_power'] == '':
            df.at[i, 'max_power'] = np.nan
        if df.at[i, 'mileage'] is not np.nan and len(re.findall(r'\d+.', df.at[i, 'mileage'])) > 0:
            if 'kg' in df.at[i, 'mileage']:
                df.at[i, 'mileage'] = float(re.findall(r'\d+.?\d*', df.at[i, 'mileage'])[0].replace(',', '')) * 1.4
            else:
                df.at[i, 'mileage'] = float(re.findall(r'\d+.?\d*', df.at[i, 'mileage'])[0].replace(',', ''))
        else:
            df.at[i, 'mileage'] = np.nan
    df['engine'] = df['engine'].str.replace(r'(\d+) CC', r'\1', regex=True).astype(float)
    df['max_power'] = df['max_power'].str.replace(r'(\d+.?\d*) bhp', r'\1', regex=True)
    df['max_power'] = df['max_power'].str.replace(r'(\d+) ', r'\1', regex=True).astype(float)

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    # filler
    df_mean = joblib.load(path_to_mean)
    df = df.fillna(df_mean)
    df['seats'] = df['seats'].astype(object)
    df['mileage'] = df['mileage'].astype(float)

    df = pd.get_dummies(df)
    df = df.reindex(columns=list(x_cols), fill_value=0)

    with open(path_to_scaler, 'rb') as file:
        scaler = pickle.load(file)
    df = scaler.transform(df)
    return df


def return_df(items: list[Item]):
    res = pd.DataFrame()
    for i in items:
        if len(res) > 0:
            res = pd.concat([res, i.return_df()], ignore_index=True)
        else:
            res = i.return_df()
    return res


# не разобрался как сделать глобальные объекты
# с помощью startup - если будет возможность -
# подскажите в комментах пожалуйста)
# @app.on_event("startup")
# def init_model():
#     with open(path_to_model, 'rb') as file:
#         model = pickle.load(file)
#     return model

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
    # make df
    inp_df = item.return_df()
    print(inp_df)
    #  preprocess data
    ready_dt = prepare_df(inp_df)
    print(ready_dt)
    # predict
    y_pred = model.predict(ready_dt)
    return y_pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
    # make df
    inp_df = return_df(items)
    #  preprocess data
    ready_df = prepare_df(inp_df)
    # predict
    y_pred = model.predict(ready_df)
    return list(y_pred)
