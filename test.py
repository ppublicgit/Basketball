import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from copy import deepcopy
import datetime
import re
from itertools import chain
import numpy as np


class BadResponse:
    def __init__(self):
        self.status_code = 404


def get_scores_url(day, month, year):
    url = "https://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate={day}%2F{month}%2F{year}".format(
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
        referer = "https://stats.nba.com/game/{game_id}/".format(
            game_id=str(game_id))
    elif requestType == "advgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/advanced".format(
            game_id=str(game_id))
    elif requestType == "tradgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/".format(
            game_id=str(game_id))
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
    scoreResponses = [get_response(
        "scores", date[0], date[1], date[2]) for date in dates]
    game_ids = [data_to_game_ids(r.json())
                for r in scoreResponses if r.status_code != 404]
    game_ids = flatten_2d(game_ids)
    columns = ["game_id", "home_flag", "team_id", "abb", "wins", "loss", "wl%",
               "asts", "rebs", "orebs", "tovs", "fga", "fg%",
               "2pa", "2p%", "3pa", "3p%", "fta", "ft%",
               "pfs", "pts", "ref1", "ref2", "ref3", "net_score", "won"]
    observations = []
    for gid in game_ids:
        obs = get_observation(gid)
        if obs is not None:
            for i in range(len(obs)):
                observations.append(obs[i])
    df = pd.DataFrame(data=observations, columns=columns)
    df.set_index(["game_id", "home_flag"], inplace=True)
    return df


def flatten_2d(unflat):
    if len(np.shape(unflat)) == 2:
        return [j for sub in unflat for j in sub]
    elif len(np.shape(unflat)) == 1:
        return unflat
    else:
        raise ValueError(
            "call to flatten_2d passed an invlaid shape (>2 or 0)")


def isHomeTeam(home_team_id, team_id):
    if home_team_id == team_id:
        return 1
    else:
        return 0


def get_observation(game_id):
    sum_response = get_response("sumgame", game_id)
    trad_response = get_response("tradgame", game_id)
    if sum_response.status_code == 404 or trad_response.status_code == 404:
        return None
    sum_data = sum_response.json()
    trad_data = trad_response.json()
    trad = trad_data["resultSets"][1]["rowSet"]
    refs = [None]*3
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
        row = [game_id, isHomeTeam(home_team, trad[i][1]), trad[i][1], trad[i][3],
               int(wins), int(losses), float(wins)/(float(wins)+float(losses)),
               int(trad[i][18]), int(trad[i][17]), int(
                   trad[i][15]), int(trad[i][21]),
               int(trad[i][7]), float(
                   trad[i][8]), None, None, int(trad[i][10]),
               float(trad[i][11]), int(trad[i][13]), float(trad[i][14]),
               int(trad[i][22]), int(trad[i][23]), refs[0], refs[1], refs[2], None, 0]
        row[13] = row[11] - row[15]
        row[14] = round(
            ((row[12]*(row[13]+row[15])-row[15]*row[16])/row[13]), 3)
        rows.append(row)
        team_pts.append(int(trad[i][23]))
    rows[0][-2] = team_pts[0] - team_pts[1]
    rows[1][-2] = team_pts[1] - team_pts[0]
    if rows[0][-2] > 0:
        rows[0][-1] = 1
    elif rows[0][-2] < 0:
        rows[1][-1] = 1
    return rows


def str_(number):
    return "{:02d}".format(number)


def generate_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    if start == end:
        return [tuple(start_date.split("-"))]
    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    date_range = [(str_(date.day), str_(date.month), str_(date.year))
                  for date in date_generated]
    return date_range


df = get_dataframe("03-09-2020", "03-09-2020")
