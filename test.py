import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from copy import deepcopy
import datetime
import re
from itertools import chain


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
    HEADERS = {
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
    elif requestType == "game_sum":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/".format(game_id=str(game_id))
    elif requestType == "advgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/advanced".format(game_id=str(game_id))
    elif requestType == "tradgame":
        game_id = args[0]
        referer = "https://stats.nba.com/game/{game_id}/".format(game_id=str(game_id))
    headers = deepcopy(HEADERS)
    headers["Referer"] = referer
    return headers


def get_response(requestType, *args):
    if requestType == "scores":
        url = get_scores_url(*args)
    elif requestType == "game_sum":
        url = get_game_sum_url(*args)
    elif requestType == "advgame":
        url = get_adv_game_url(*args)
    elif requestType == "tradgame":
        url = get_traditional_game_url(*args)
    headers = get_request_header(requestType, *args)
    response = requests.get(url, headers=headers, timeout=30)
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
    scoreResponses = [get_response("scores", date) for date in dates]
    game_ids = [data_to_game_ids(r.json()) for r in scoreResponses if r.status_code != 404]
    flattened_games = [j for sub in game_ids for j in sub]



def str_(number):
    return "{:02d}".format(number)


def generate_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    date_range = [(str_(date.day), str_(date.month), str_(date.year)) for date in date_generated]
    return date_range


breakpoint()
