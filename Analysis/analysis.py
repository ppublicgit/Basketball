import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def get_single_season_df(year, datapath):
    return pd.read_csv(glob.glob(os.path.join(datapath, f"*_game*_{year}_*"))[0])


def split_game_ids(game_ids, train_size, val_size, test_size):

    num_games = len(game_ids)
    training = int(train_size*num_games)
    if val_size != 0:
        validation = int(val_size*num_games) + training
    else:
        validation = training + 1

    training_games = game_ids[:training]
    validation_games = game_ids[training:validation]
    testing_games = game_ids[validation:]

    return training_games, validation_games, testing_games


def split_dfs(df, game_ids, train_size, val_size, test_size, seed=42):
    np.random.seed(seed)
    np.random.shuffle(game_ids)

    training_games, validation_games, testing_games = split_game_ids(game_ids,
                                                                     train_size,
                                                                     val_size,
                                                                     test_size)

    training_df = df.loc[training_games]
    validation_df = df.loc[validation_games]
    testing_df = df.loc[testing_games]

    return training_df, validation_df, testing_df


def get_x_y(df, cols, outtype="Classification"):


    subset_home = df.query("home_flag == '1'").droplevel(1).loc[:, cols]
    subset_away = df.query("home_flag == '0'").droplevel(1).loc[:, cols]
    subset = subset_home.subtract(subset_away)
    subset["net_score"] = subset["net_score"] / 2
    subset["won"] = (subset["won"] + 1)// 2


    X = subset.iloc[:, 1:-2].to_numpy()
    if outtype == "Classification":
        y = subset.iloc[:, -1].to_numpy()
    else:
        y = subset.iloc[:, -2].to_numpy()

    return X, y


def get_team_df(datapath):
    return pd.read_csv(glob.glob(os.path.join(datapath, "*team*"))[0])


def get_predict_x_y(testing_df, team_df, features):
    X = []
    for game_id in np.unique([index[0] for index in testing_df.index.values]):
        home_team = testing_df.loc[game_id, 1]["team_id"]
        away_team = testing_df.loc[game_id, 0]["team_id"]
        home_team_df = team_df.loc[home_team, features].drop("wl%")
        away_team_df = team_df.loc[away_team, features].drop("wl%")
        data_point = home_team_df.subtract(away_team_df)
        X.append(data_point.to_numpy())

    y = testing_df.query("home_flag == '1'").loc[:, "won"].values
    return X, y


def test_control_score(testing_df, team_df):
    predicted = []
    for game_id in np.unique([index[0] for index in testing_df.index.values]):
        home_team = testing_df.loc[game_id, 1]["team_id"]
        away_team = testing_df.loc[game_id, 0]["team_id"]
        wl_comp = team_df.loc[home_team, "wl%"] - team_df.loc[home_team, "wl%"]
        if wl_comp < 0:
            predicted.append(0)
        else:
            predicted.append(1)
    Testy = testing_df.query("home_flag == '1'").loc[:, "won"].values
    correct = 0
    for i in range(len(predicted)):
        if Testy[i] == predicted[i]:
            correct += 1
    return correct/len(predicted)


def main(year, path):
    df = get_single_season_df(year, path)

    features = list(df.columns[6:-6])
    regression = df.columns[-2]
    label = df.columns[-1]
    cols = features + [regression] + [label]

    df2 = df.set_index(["game_id", "home_flag"])

    game_ids = np.array(df2.index.levels[0])

    train_df, val_df, test_df = split_dfs(df2, game_ids, 0.75, 0, 0.25)

    X_train, y_train = get_x_y(train_df, cols)
    X_test, y_test = get_x_y(test_df, cols)

    team_df = get_team_df(path)
    team_df_year = team_df.loc[team_df["season"] == YEARS[str(year)]]
    team_df_year.set_index("team_id", inplace=True)
    team_df_year = team_df_year.loc[:, features]

    X_predict, y_predict = get_predict_x_y(test_df, team_df_year, features)


    clf = LogisticRegression(solver="newton-cg")
    clf = SVC(kernel="rbf", gamma="scale")
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    predict_score = clf.score(X_predict, y_predict)
    control_score = test_control_score(test_df, team_df_year)

    #print("")
    #pca_ = PCA(n_components=8)
    #pca_ = pca_.fit(X_train)
    #X_train_pca = pca_.transform(X_train)
    #X_test_pca = pca_.transform(X_test)
    #
    #clf = LogisticRegression(solver="newton-cg")
    #clf = SVC(kernel="rbf", gamma="scale")
    ##clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    #clf.fit(X_train_pca, y_train)
    #print(clf.score(X_train_pca, y_train))
    #print(clf.score(X_test_pca, y_test))
    #print(test_control_score(test_df, team_df_year))

    return train_score, test_score, predict_score, control_score


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), "Data/SuccessfulScrape")
    year = 2010
    YEARS = {"2010":"2009-10",
             "2011":"2010-11",
             "2012":"2011-12",
             "2013":"2012-13",
             "2014":"2013-14",
             "2015":"2014-15",
             "2016":"2015-16",
             "2017":"2016-17",
             "2018":"2017-18"}


    train_scores, test_scores, predict_scores, control_scores = [], [] ,[], []
    for year in range(2010, 2019):
        tr_sc, te_sc, pr_sc, co_sc = main(year, path)
        train_scores.append(tr_sc)
        test_scores.append(te_sc)
        predict_scores.append(pr_sc)
        control_scores.append(co_sc)

    train_mean = np.mean(train_scores)
    train_std = np.std(train_scores)

    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)

    predict_mean = np.mean(predict_scores)
    predict_std = np.std(predict_scores)

    control_mean = np.mean(control_scores)
    control_std = np.std(control_scores)

    print("TRAIN")
    print(f"{train_mean*100:.02f}")
    print(f"{train_std*100:.02f}")
    print("TEST")
    print(f"{test_mean*100:.02f}")
    print(f"{test_std*100:.02f}")
    print("PREDICT")
    print(f"{predict_mean*100:.02f}")
    print(f"{predict_std*100:.02f}")
    print("CONTROL")
    print(f"{control_mean*100:.02f}")
    print(f"{control_std*100:.02f}")
