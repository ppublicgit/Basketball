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
         "2018":"2017-18",
         "2019":"2018-19",
         "2020":"2019-20"}

def get_single_season_df(datapath, year):
    return pd.read_csv(glob.glob(os.path.join(datapath, f"*_game*_{year}_*"))[0])

def train_test_split(df, train_size, validation_size, test_size, **kwargs):

    def almost_equals(left, right):
        return abs(left-right) < 0.001

    seed = kwargs.get("seed", 42)

    if almost_equals(train_size + test_size + validation_size, 0):
        raise ValueError("train_size + test_size + validation_size does not equal 1")

    game_ids = np.unique(df["game_id"].values)

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
    validation_df = df.loc[validation_games]

    return training_df, validation_df, testing_df

def transform_df(df, **kwargs):
    features = kwargs.get("features", list(df.columns[4:-6]))
    regression = kwargs.get("regresssion", df.columns[-2])
    label = kwargs.get("label", df.columns[-1])
    cols = features + [regression] + [label]

    df_subset_home = df.query("home_flag == '1'").droplevel(1).loc[:, cols]
    df_subset_away = df.query("home_flag == '0'").droplevel(1).loc[:, cols]
    df_subset = df_subset_home.subtract(df_subset_away)
    df_subset["net_score"] = df_subset["net_score"] / 2
    df_subset["won"] = (df_subset["won"] + 1)// 2

    return df_subset


def get_team_df(datapath, **kwargs):
    year = kwargs.get("year", None)

    if str(year) not in YEARS.keys():
        raise ValueError(f"Season {year} not supported. Only season {YEARS.keys()} are supported.")

    df = pd.read_csv(glob.glob(os.path.join(datapath, "*team*"))[0])

    features = kwargs.get("features", list(df.columns[4:-1]))

    if year is not None:
        df = df.loc[df["season"] == YEARS[str(year)]]
        df.set_index("team_id", inplace=True)
        df = df.loc[:, features]

    return df


def get_test_input(testing_df, team_df, **kwargs):

    features = kwargs.get("features", list(team_df.columns))

    X = []
    for game_id in np.unique([index[0] for index in testing_df.index.values]):
        home_team = testing_df.loc[game_id, 1]["team_id"]
        away_team = testing_df.loc[game_id, 0]["team_id"]
        home_team_df = team_df.loc[home_team, features].drop("wl%")
        away_team_df = team_df.loc[away_team, features].drop("wl%")
        data_point = home_team_df.subtract(away_team_df)
        X.append(data_point.to_numpy())
    return X


def get_control_score(season_df, team_df):
    correct, incorrect = 0, 0
    for game_id in np.unique([index[0] for index in season_df.index.values]):
        home_team = season_df.loc[game_id, 1]["team_id"]
        away_team = season_df.loc[game_id, 0]["team_id"]
        wl_comp = team_df.loc[home_team, "wl%"] - team_df.loc[away_team, "wl%"]
        if (wl_comp < 0 and season_df.loc[game_id, 1]["won"] == 0) or \
           (wl_comp >= 0 and season_df.loc[game_id, 1]["won"] == 1):
            correct += 1
        else:
            incorrect += 1
    return correct/(correct + incorrect)


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    YEAR = 2016
    DATAPATH = os.path.join(os.path.dirname(os.getcwd()), "Data/SuccessfulScrape")

    df = get_single_season_df(DATAPATH, YEAR)

    train_df, validate_df, test_df = train_test_split(df, 0.7, 0.1, 0.2)

    train_df = transform_df(train_df)
    validate_df = transform_df(validate_df)

    clf = LogisticRegression(solver="newton-cg")

    X = train_df.iloc[:, 1:-2 ].to_numpy()
    y = train_df.iloc[:, -1].to_numpy()

    VX = validate_df.iloc[:, 1:-2 ].to_numpy()
    Vy = validate_df.iloc[:, -1].to_numpy()

    clf.fit(X, y)

    print(f"Test Score:\t\t{clf.score(X, y)}")
    print(f"Validate Score:\t{clf.score(VX, Vy)}")

    team_df = get_team_df(DATAPATH, year=YEAR)

    TestX = get_test_input(test_df, team_df)

    Testy = test_df.query("home_flag == '1'").loc[:, "won"].values

    print(f"Test Score:\t\t{clf.score(TestX, Testy)}")

    print(f"Control Score:\t{get_control_score(test_df, team_df)}")
