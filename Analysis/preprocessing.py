import os
import glob
import re
import numpy as np
import pandas as pd


YEARS = {"2010":"2009-10",
         "2011":"2010-11",
         "2012":"2011-12",
         "2013":"2012-13",
         "2014":"2013-14",
         "2015":"2014-15",
         "2016":"2015-16",
         "2017":"2016-17",
         "2018":"2017-18"}

def get_single_season_df(year, datapath):
    return pd.read_csv(glob.glob(os.path.join(datapath, f"*_game*_{year}_*"))[0])

def train_test_split(df, train_size, test_size, **kwargs):

    seed = kwargs.get("seed", 42)
    validation_size = kwargs.get("validation_size", 0)

    if almostEquals(train_size + test_size + validation_size, 0):
        raise ValueError("train_size + test_size + validation_size does not equal 1")

    game_ids = np.unique(df["game_ids"].values)

    np.random.seed(seed)
    np.random.shuffle(game_ids)

    num_games = len(game_ids)
    training = int(train_size*num_games)
    validation = int(validation_size*num_games) + training

    training_games = game_ids[:training]
    validation_games = game_ids[training:validation]
    testing_games = game_ids[validation:]

    df = df.set_index(["game_id", "home_flag"])

    training_df = df.loc[training_games]
    testing_df = df.loc[testing_games]
    if validation_size > 0:
        validation_df = df.loc[validation_games]
        return training_df, validation_df, testing_df

    return training_df, testing_df

def transform_df(df, **kwargs):
    features = kwargs.get("features", list(df.columns[6:-6]))
    regression = kwargs.get("regresssion", df.columns[-2])
    label = kwargs.get("label", df.columns[-1])
    cols = features + [regression] + [label]
