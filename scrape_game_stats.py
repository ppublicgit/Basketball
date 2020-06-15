import requests
import pandas as pd
from time import sleep
import time
from copy import deepcopy
import datetime
import re
from itertools import chain
import numpy as np
import os


TODAY = str(datetime.date.today())


class BadResponse:
    def __init__(self):
        self.status_code = 404


def get_scores_url(day, month, year):
    url = "https://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate={month}%2F{day}%2F{year}".format(
        day=str(day), month=str(month), year=str(year)
    )
    return url


def get_game_sum_url(game_id):
    url = "https://stats.nba.com/stats/boxscoresummaryv2?GameID={game_id}".format(
        game_id=str(game_id)
    )
    return url


def get_adv_game_url(game_id):
    url = "https://stats.nba.com/stats/boxscoreadvancedv2?EndPeriod=10&EndRange=34800&GameID={game_id}&RangeType=0&Season=2019-20&SeasonType=Regular+Season&StartPeriod=1&StartRange=0".format(
        game_id=str(game_id)
    )
    return url


def get_traditional_game_url(game_id):
    url = "https://stats.nba.com/stats/boxscoretraditionalv2?EndPeriod=10&EndRange=34800&GameID={game_id}&RangeType=0&Season=2019-20&SeasonType=Regular+Season&StartPeriod=1&StartRange=0".format(
        game_id=str(game_id)
    )
    return url


def get_request_header(requestType, *args):
    headers = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "DNT": "1",
        "Connection": "keep-alive",
        "Referer": "https://stats.nba.com/scores/03/10/2020",
    }
    if requestType == "scores":
        day, month, year = args[0], args[1], args[2]
        referer = "https://stats.nba.com/scores/{day}/{month}/{year}".format(
            day=str(day), month=str(month), year=str(year)
        )
    elif requestType == "sumgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/".format(game_id=str(game_id))
    elif requestType == "advgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/advanced".format(
            game_id=str(game_id)
        )
    elif requestType == "tradgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/".format(game_id=str(game_id))
    headers["Referer"] = referer
    return headers


def get_response(requestType, *args):
    if requestType == "scores":
        url = get_scores_url(*args)
    elif requestType == "sumgame":
        url = get_game_sum_url(*args)
    elif requestType == "advgame":
        url = get_adv_game_url(*args)
    elif requestType == "tradgame":
        url = get_traditional_game_url(*args)
    headers = get_request_header(requestType, *args)
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except:
        return BadResponse()
    sleep(15)
    return response


def generate_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    if start == end:
        return [tuple(start_date.split("-"))]
    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end - start).days)
    ]
    date_range = [
        (str_(date.day), str_(date.month), str_(date.year)) for date in date_generated
    ]
    return date_range


def get_game_ids(date_range):
    def check_reg_season(gid):
        if isinstance(gid, str):
            for i in range(len(gid)):
                if gid[i] == "0":
                    continue
                elif gid[i] == "1":
                    return False
                elif gid[i] == "2":
                    return True
        else:
            for i in range(len(gid[0])):
                if gid[i] == "0":
                    continue
                elif gid[i] == "1":
                    return False
                elif gid[i] == "2":
                    return True

    def get_game_id_bounds(index, direction):
        if direction == "forward" and index < len(date_range):
            test = date_range[index]
            scoreResponse = get_response("scores", test[0], test[1], test[2])
            if scoreResponse.status_code == 200:
                start_ids = data_to_game_ids(scoreResponse.json())
                if len(start_ids) > 0 and check_reg_season(start_ids[0]):
                    return start_ids
                else:
                    return get_game_id_bounds(index+1, "forward")
            else:
                return get_game_id_bounds(index+1, "forward")
        elif direction == "backward" and index >= 0:
            test = date_range[index]
            scoreResponse = get_response("scores", test[0], test[1], test[2])
            if scoreResponse.status_code == 200:
                end_ids = data_to_game_ids(scoreResponse.json())
                if len(end_ids) > 0:
                    return end_ids
                else:
                    return get_game_id_bounds(index-1, "backward")
            else:
                return get_game_id_bounds(index-1, "backward")
        else:
            return None

    def create_game_ids_list(start, end):
        if isinstance(start, list):
            start = start[0]
        if isinstance(end, list):
            end = end[-1]
        game_id_len = len(start)
        start_int, end_int = int(start), int(end)
        game_ids = []
        while start_int != end_int:
            new_game_id = str_(start_int, game_id_len)
            game_ids.append(new_game_id)
            start_int += 1
        return game_ids

    start_id = get_game_id_bounds(0, "forward")
    end_id = get_game_id_bounds(len(date_range)-1, "backward")
    if start_id is None and end_id is None:
        breakpoint()
        return None
    elif start_id is None:
        breakpoint()
        return end_id
    elif end_id is None:
        breakpoint()
        return start_id
    elif start_id == end_id and isinstance(start_id, str):
        breakpoint()
        return [start_id]
    elif start_id == end_id:
        breakpoint()
        return start_id
    else:
        return create_game_ids_list(start_id, end_id)


def data_to_game_ids(data):
    games = data["resultSets"][0]["rowSet"]
    game_ids = [games[i][2] for i in range(len(games))]
    return game_ids


def get_dataframe(start_date, end_date):
    if not bool(re.match(r"[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]", start_date)):
        raise ValueError("Invalid start date input")
    if not bool(re.match(r"[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]", end_date)):
        raise ValueError("Invalid end date input")
    dates = generate_dates(start_date, end_date)
    game_ids = get_game_ids(dates)
    columns = [
        "game_id",
        "home_flag",
        "team_id",
        "abb",
        "wins",
        "loss",
        "wl%",
        "asts",
        "rebs",
        "orebs",
        "tovs",
        "fga",
        "fg%",
        "2pa",
        "2p%",
        "3pa",
        "3p%",
        "fta",
        "ft%",
        "pfs",
        "pts",
        "ref1",
        "ref2",
        "ref3",
        "net_score",
        "won",
    ]
    observations = []
    counter = 0
    start = time.time()
    if game_ids is None:
        raise ValueError("Error with {start_date} {end_date}")
        return None
    for gid in game_ids:
        if counter % (len(game_ids) // 10) == 0:
            print(f"{counter/len(game_ids)} % complete -- {time.time()-start} seconds")
        obs = get_observation(gid)
        if obs is not None:
            for i in range(len(obs)):
                observations.append(obs[i])
        counter += 1
    if len(observations) > 0:
        df = pd.DataFrame(data=observations, columns=columns)
        df.set_index(["game_id", "home_flag"], inplace=True)
        df.sort_index(inplace=True)
        return df
    else:
        return None


def isHomeTeam(home_team_id, team_id):
    if home_team_id == team_id:
        return 1
    else:
        return 0


def log(message, path=os.path.join(os.getcwd(), "Data")):
    with open(os.path.join(path, f"log_{TODAY}.txt"), "a") as f:
        f.write(message)
        f.write("\n")


def get_observation(game_id):
    sum_response = get_response("sumgame", game_id)
    trad_response = get_response("tradgame", game_id)
    if sum_response.status_code != 200 or trad_response.status_code != 200:
        log(f"bad game id {game_id}")
        return None
    sum_data = sum_response.json()
    trad_data = trad_response.json()
    trad = trad_data["resultSets"][1]["rowSet"]
    refs = [None] * 3
    for i in range(3):
        try:
            refs[i] = sum_data["resultSets"][2]["rowSet"][i][0]
        except:
            refs[i] = "None"
    home_team = sum_data["resultSets"][0]["rowSet"][0][6]
    rows = []
    team_pts = []
    for i in range(2):
        wins, losses = sum_data["resultSets"][5]["rowSet"][i][7].split("-")
        row = [
            game_id,
            isHomeTeam(home_team, trad[i][1]),
            trad[i][1],
            trad[i][3],
            int(wins),
            int(losses),
            round(float(wins) / (float(wins) + float(losses)), 3),
            int(trad[i][18]),
            int(trad[i][17]),
            int(trad[i][15]),
            int(trad[i][21]),
            int(trad[i][7]),
            float(trad[i][8]),
            None,
            None,
            int(trad[i][10]),
            float(trad[i][11]),
            int(trad[i][13]),
            float(trad[i][14]),
            int(trad[i][22]),
            int(trad[i][23]),
            refs[0],
            refs[1],
            refs[2],
            None,
            0,
        ]
        row[13] = row[11] - row[15]
        row[14] = round(
            ((row[12] * (row[13] + row[15]) - row[15] * row[16]) / row[13]), 3
        )
        rows.append(row)
        team_pts.append(int(trad[i][23]))
    rows[0][-2] = team_pts[0] - team_pts[1]
    rows[1][-2] = team_pts[1] - team_pts[0]
    if rows[0][-2] > 0:
        rows[0][-1] = 1
    elif rows[0][-2] < 0:
        rows[1][-1] = 1
    return rows


def str_(number, length=2):
    return str(number).zfill(length)


def get_data_inputs_frame(df):
    columns = [
        "wl%",
        "asts",
        "rebs",
        "orebs",
        "tovs",
        "fga",
        "fg%",
        "3pa",
        "3p%",
        "fta",
        "ft%",
        "pfs",
        "net_score",
        "won",
    ]
    temp_dict = {}
    for col in columns:
        temp_dict[col] = []
    for i in range(0, len(df), 2):
        away = df.iloc[i]
        home = df.iloc[i + 1]
        for col in columns:
            new_val = home[col] - away[col]
            if col == "net_score":
                new_val /= 2
            temp_dict[col].append(new_val)
    return pd.DataFrame(data=temp_dict, columns=columns)


def split_learn_df(df, classtype="discrete"):
    inputs = df.iloc[:, :-2].to_numpy()
    features = df.columns[:-2]
    if classtype == "discrete":
        classes = df.iloc[:, -1].values
    elif classtype == "continuous":
        classes = df.iloc[:, -2].values
    else:
        raise ValueError("Invalid class type")
    return inputs, features, classes


if __name__ == "__main__":
    seasons = [*range(2020, 2021, 1)]
    for season in seasons:
        log(f"season: {season}")
        print(season)
        start_date = "01-12-" + str(season - 1)
        if season == 2012:
            start_date = "01-01-2012"  # lockout shortened season
        end_date = "01-04-" + str(season)
        if season == 2020:
            end_data = "12-03-" + str(season)  # covid shortened season
        df = get_dataframe(start_date, end_date)
        if df is not None:
            df.to_csv(f"Data/season_game_info_df_{season}_{TODAY}.csv")

    # learning_df = get_data_learn_frame(df)
    # inputs, features, classes = split_learn_df(learning_df)
