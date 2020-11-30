import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import time

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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


def split_dfs(df, game_ids, train_size, val_size, test_size, seed=42, ff=False):
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


def get_x_y(df, cols, outtype="Classification", ff=False):
    if ff:
        X = df.iloc[:, 2:-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    else:
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


def get_predict_x_y(testing_df, team_df, features, ff=False):
    if ff:
        X = []
        for game_id in np.unique(testing_df.index.values):
            home_team = testing_df.loc[game_id, "home_team"]
            away_team = testing_df.loc[game_id, "away_team"]
            game_df = pd.DataFrame()
            game_df = game_df.append(team_df.loc[home_team, :], ignore_index=True)
            game_df = game_df.append(team_df.loc[away_team, :], ignore_index=True)
            game_df["game_id"] = game_id
            game_df["home_flag"] = [1, 0]
            game_df["won"] = [testing_df.loc[game_id, "won"], (testing_df.loc[game_id, "won"] + 1)%2]
            game_df["team_id"] = [home_team, away_team]
            data_point_df = fourfactor(game_df)
            data_point = data_point_df.iloc[:, 3:-1].to_numpy()
            data_point = data_point.reshape(data_point.shape[1])
            X.append(data_point)

        y = testing_df["won"].values
    else:
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


def test_control_score(testing_df, team_df, ff=False):
    predicted = []
    if ff:
        indexes = np.unique(testing_df.index.values)
    else:
        indexes = np.unique([index[0] for index in testing_df.index.values])

    for game_id in indexes:
        if ff:
            home_team = testing_df.loc[game_id, "home_team"]
            away_team = testing_df.loc[game_id, "away_team"]
        else:
            home_team = testing_df.loc[game_id, 1]["team_id"]
            away_team = testing_df.loc[game_id, 0]["team_id"]
        wl_comp = team_df.loc[home_team, "wl%"] - team_df.loc[away_team, "wl%"]
        if wl_comp < 0:
            predicted.append(0)
        else:
            predicted.append(1)
    if ff:
        Testy = testing_df["won"].values
    else:
        Testy = testing_df.query("home_flag == '1'").loc[:, "won"].values
    correct = 0
    for i in range(len(predicted)):
        if Testy[i] == predicted[i]:
            correct += 1
    return correct/len(predicted)



def corr(df, ff=False):
    if not ff:
        title = f"Correlation Matrix\nRaw Statistics"
        corr_df = df.loc[:, ["wl%", "asts", "rebs", "orebs",
                             "tovs", "fga", "fg%", "2pa",
                             "2p%", "3pa", "3p%", "fta",
                             "ft%", "pfs", "pts", "won"]]
    else:
        title = f"Correlation Matrix\nFour Factor Statistics"
        corr_df = df.iloc[:, 2:]
    corr = corr_df.corr()
    plt.figure()
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 18})
    sns.heatmap(corr, xticklabels=corr_df.columns, yticklabels=corr_df.columns)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show(block=False)
    return


def fourfactor(df):

    temp_df = pd.DataFrame()
    temp_df["game_id"] = df["game_id"]
    temp_df["home_flag"] = df["home_flag"]
    temp_df["Shooting"] = ((df["2p%"] * df["2pa"] + 0.5 * df["3p%"] * df["3pa"]) / df["fga"])
    temp_df["Turnovers"] = (df["tovs"] / (df["fga"] + 0.44 * df["fta"] + df["tovs"]) )
    temp_df["Free Throws"] = df["fta"] * df["ft%"] / df["fga"]

    ff_df = pd.DataFrame()
    ff_df["game_id"] = temp_df.loc[(temp_df["home_flag"] == 1), "game_id"]
    ff_df["home_team"] = df.loc[(df["home_flag"] == 1), "team_id"]
    ff_df["away_team"] = df.loc[(df["home_flag"] == 0), "team_id"].values
    ff_df["sho off"] = temp_df.loc[(temp_df["home_flag"] == 1), "Shooting"]
    ff_df["sho def"] = temp_df.loc[(temp_df["home_flag"] == 0), "Shooting"].values
    ff_df["tov off"] = temp_df.loc[(temp_df["home_flag"] == 1), "Turnovers"]
    ff_df["tov ofF"] = temp_df.loc[(temp_df["home_flag"] == 0), "Turnovers"].values
    ff_df["ft off"] = temp_df.loc[(temp_df["home_flag"] == 1), "Free Throws"]
    ff_df["ft def"] = temp_df.loc[(temp_df["home_flag"] == 0), "Free Throws"].values

    ff_df["reb off"] = \
        df.loc[(df["home_flag"] == 1), "orebs"].values / \
        (df.loc[(df["home_flag"] == 0), "rebs"].values - \
         df.loc[(df["home_flag"] == 0), "orebs"].values + \
         df.loc[(df["home_flag"] == 1), "orebs"].values)

    ff_df["reb def"] = \
        (df.loc[(df["home_flag"] == 1), "rebs"].values - \
         df.loc[(df["home_flag"] == 1), "orebs"].values) / \
        (df.loc[(df["home_flag"] == 1), "rebs"].values - \
         df.loc[(df["home_flag"] == 1), "orebs"].values + \
         df.loc[(df["home_flag"] == 0), "orebs"].values)

    ff_df["won"] = df.loc[(df["home_flag"] == 1), "won"]

    return ff_df


def main(year, path, plot_corr, classifier):
    start = time.time()
    clfs = {"RF": RandomForestClassifier(n_estimators=25),
            "SVM": SVC(kernel="rbf", gamma="scale"),
            "LogR": LogisticRegression(solver="newton-cg")}

    df = get_single_season_df(year, path)

    features = list(df.columns[6:-6])
    regression = df.columns[-2]
    label = df.columns[-1]
    cols = features + [regression] + [label]

    df2 = df.set_index(["game_id", "home_flag"])

    if plot_corr:
        corr(df2)

    game_ids = np.array(df2.index.levels[0])

    train_df, val_df, test_df = split_dfs(df2, game_ids, 0.75, 0, 0.25)

    X_train, y_train = get_x_y(train_df, cols)
    X_test, y_test = get_x_y(test_df, cols)

    team_df = get_team_df(path)
    team_df_year = team_df.loc[team_df["season"] == YEARS[str(year)]]
    team_df_year.set_index("team_id", inplace=True)
    team_df_year = team_df_year.loc[:, features]

    X_predict, y_predict = get_predict_x_y(test_df, team_df_year, features)

    clf = clfs[classifier]
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    predict_score = clf.score(X_predict, y_predict)
    control_score = test_control_score(test_df, team_df_year)

    finish = time.time() - start

    return train_score, test_score, predict_score, control_score, finish


def main_ff(year, path, plot_corr, classifer):
    start = time.time()
    clfs = {"RF": RandomForestClassifier(n_estimators=25),
            "SVM": SVC(kernel="rbf", gamma="scale"),
            "LogR": LogisticRegression(solver="newton-cg")}

    df = get_single_season_df(year, path)

    features = list(df.columns[6:-6])
    regression = df.columns[-2]
    label = df.columns[-1]
    cols = features + [regression] + [label]

    ff_df = fourfactor(df)

    ff_df = ff_df.set_index("game_id")

    if plot_corr:
        corr(ff_df, ff=True)

    game_ids = np.array(ff_df.index.values)

    train_df, val_df, test_df = split_dfs(ff_df, game_ids, 0.75, 0, 0.25, ff=True)
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    X_train, y_train = get_x_y(train_df, cols, ff=True)
    X_test, y_test = get_x_y(test_df, cols, ff=True)

    team_df = get_team_df(path)
    team_df_year = team_df.loc[team_df["season"] == YEARS[str(year)]]
    team_df_year.set_index("team_id", inplace=True)
    team_df_year = team_df_year.loc[:, features]

    X_predict, y_predict = get_predict_x_y(test_df, team_df_year, features, ff=True)

    clf = clfs[classifier]
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    predict_score = clf.score(X_predict, y_predict)
    control_score = test_control_score(test_df, team_df_year, ff=True)

    finish = time.time() - start
    return train_score, test_score, predict_score, control_score, finish


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), "Data/SuccessfulScrape")
    classifiers = ["RF", "SVM", "LogR"]
    years = [*range(2010, 2019)]
    df = pd.DataFrame(columns=["Classifier", "Year", "Train",
                               "Test", "Predict", "Control", "Runtime"])
    for classifier in classifiers:
        print(classifier)
        for year in years:
            print(year)
            if year == 2010 and classifier == "RF":
                plot_corr = True
            else:
                plot_corr = False
            tr_sc, te_sc, pr_sc, co_sc, finish = main(year, path, plot_corr, classifier)
            row = {"Classifier": classifier,
                   "Year": year,
                   "Train": tr_sc,
                   "Test": te_sc,
                   "Predict": pr_sc,
                   "Control": co_sc,
                   "Runtime": finish
                   }
            df = df.append(row, ignore_index=True)

    df = df.set_index(pd.MultiIndex.from_product([classifiers, years],
                                                 names=["Classifier", "Year"]))
    df = df.drop(columns=["Classifier"])
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 18})
    ax = df.boxplot(by="Classifier", column=["Train", "Test", "Predict", "Control"],
                    layout=(2,2))
    ax[0, 0].set_xlabel("Classifier")
    ax[0, 0].set_ylabel("Performance Accuracy")
    ax[0, 1].set_xlabel("Classifier")
    ax[0, 1].set_ylabel("Performance Accuracy")
    ax[1, 0].set_xlabel("Classifier")
    ax[1, 0].set_ylabel("Performance Accuracy")
    ax[1, 1].set_xlabel("Classifier")
    ax[1, 1].set_ylabel("Performance Accuracy")
    plt.show(block=False)

    runtime_df = df.groupby(by="Classifier").mean()
    print(runtime_df["Runtime"])

    input("Press Enter to exit...")
