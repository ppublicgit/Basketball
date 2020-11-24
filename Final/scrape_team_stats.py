import requests
import pandas as pd
from time import sleep
from copy import deepcopy
import datetime
import re
from itertools import chain
import numpy as np


TODAY = str(datetime.date.today())


def get_trad_team_stats_url(season):
    url = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=".format(
        season=season
    )
    return url


def get_team_request_header(requestType):
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
    if requestType == "trad":
        referer = "https://stats.nba.com/teams/traditional/"
    elif requestType == "adv":
        referer = ""
    headers["Referer"] = referer
    return headers


def get_team_response(requestType, season):
    if requestType == "trad":
        url = get_trad_team_stats_url(season)
    headers = get_team_request_header(requestType)
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except:
        return BadResponse()
    sleep(15)
    return response


def get_season_data(season):
    season_response = get_team_response("trad", season)
    if season_response.status_code != 200:
        return None
    season_response_data = season_response.json()
    ssn = season_response_data["resultSets"][0]["rowSet"]
    rows = []
    for teamIndex in range(len(ssn)):
        row = [
            season,
            ssn[teamIndex][0],
            int(ssn[teamIndex][3]),
            int(ssn[teamIndex][4]),
            float(ssn[teamIndex][5]),
            int(ssn[teamIndex][19]),
            int(ssn[teamIndex][18]),
            int(ssn[teamIndex][16]),
            int(ssn[teamIndex][20]),
            int(ssn[teamIndex][8]),
            float(ssn[teamIndex][9]),
            None,
            None,
            int(ssn[teamIndex][11]),
            float(ssn[teamIndex][12]),
            int(ssn[teamIndex][14]),
            float(ssn[teamIndex][15]),
            int(ssn[teamIndex][24]),
            int(ssn[teamIndex][26]),
        ]
        row[11] = row[9] - row[13]
        row[12] = round(
            (row[10] * (row[11] + row[13]) - row[13] * row[14]) / row[11], 3
        )
        rows.append(row)
    return rows


def convert_year_to_season(year):
    if len(str(year)) != 4 or (year < 1973 or year > 2020):
        raise ValueError("Invalid year input. Must be between 1973 and 2020")
    return f"{year-1}-{str(year)[-2:]}"


def generate_seasons(start, end):
    return [convert_year_to_season(year) for year in range(start, end + 1, 1)]


def get_team_dataframe(start_season, end_season):
    seasons = generate_seasons(start_season, end_season)
    columns = [
        "season",
        "team_id",
        "wins",
        "losses",
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
    ]
    data = []
    for season in seasons:
        season_data = get_season_data(season)
        if season_data is not None:
            for team_data in season_data:
                data.append(team_data)
    df = pd.DataFrame(data=data, columns=columns)
    df.set_index(["season", "team_id"], inplace=True)
    df.sort_index(inplace=True)
    return df


if __name__ == "__main__":
    FIRST_SEASON = 2010
    FINAL_SEASON = 2020

    df = get_team_dataframe(FIRST_SEASON, FINAL_SEASON)

    df.to_csv(f"Data/season_team_info_df_{FIRST_SEASON}_to_{FINAL_SEASON}_{TODAY}.csv")
