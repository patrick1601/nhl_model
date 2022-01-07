#%%
from bs4 import BeautifulSoup
import datetime as dt
import json
import nhl_scraper_1
import numpy as np
import pandas as pd
import pickle
import requests
from typing import List
import sys
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import lxml
from elote import EloCompetitor
from elote import GlickoCompetitor
from trueskill import Rating, quality, rate
#%%
def convert_minutes(min):
    """
    convert goalie string minutes in form xx:xx to just minutes
    ...
    Parameters
    ----------
    min: str
        minutes string in form xx:xx
    Returns
    -------
    minutes_played: float
        minutes string converted to numerical minutes
    """
    try:
        split_min = min.split(':')
        minutes = int(split_min[0])
        seconds = int(split_min[1])
        minutes_played = minutes + seconds/60
        return minutes_played
    except AttributeError:
        pass
#%%
def remove_duplicates(x: list) -> list:
    """
    takes a list and removes duplicates from that list
    ...
    Parameters
    ----------
    x: list
        list from which duplicates will be removed
    Returns
    -------
    list
        list with duplicates removed
    """
    return list(dict.fromkeys(x))
#%% function to get starting goalies
def get_starting_goalies(home_abv: str, away_abv: str, date: str) -> (str, str):
    """
    scrapes starting goaltenders from dailyfaceoff.com for the specified date and teams
    ...
    Parameters
    ----------
    home_abv: str
        abbreviation for home team
    away_abv: str
        abbreviation for away team
    date: str
        string for which we want to retrieve starting goalies (ex. '01-13-2021')
    Returns
    -------
    home_goalie: str
        home goalie name
    away_goalie: str
        away goalie name
    """

    # First define a dictionary to translate team abbreviations in our df to the team names used on daily
    # faceoff
    team_translations = {'MIN':'Minnesota Wild','TOR':'Toronto Maple Leafs',
                         'PIT':'Pittsburgh Penguins', 'COL':'Colorado Avalanche',
                         'EDM':'Edmonton Oilers', 'CAR':'Carolina Hurricanes',
                         'CBJ':'Columbus Blue Jackets', 'NJD':'New Jersey Devils',
                         'DET':'Detroit Red Wings', 'OTT':'Ottawa Senators',
                         'BOS':'Boston Bruins', 'SJS':'San Jose Sharks',
                         'BUF':'Buffalo Sabres','NYI':'New York Islanders',
                         'WSH':'Washington Capitals','TBL':'Tampa Bay Lightning',
                         'STL':'St Louis Blues', 'NSH':'Nashville Predators',
                         'CHI':'Chicago Blackhawks', 'VAN':'Vancouver Canucks',
                         'CGY':'Calgary Flames', 'PHI':'Philadelphia Flyers',
                         'LAK':'Los Angeles Kings', 'MTL':'Montreal Canadiens',
                         'ANA':'Anaheim Ducks', 'DAL':'Dallas Stars',
                         'NYR':'New York Rangers', 'FLA':'Florida Panthers',
                         'WPG':'Winnipeg Jets', 'ARI':'Arizona Coyotes',
                         'VGK':'Vegas Golden Knights'}

    home_team = team_translations[home_abv]
    away_team = team_translations[away_abv]

    url = f'https://www.dailyfaceoff.com/starting-goalies/{date}'

    # Need headers as daily faceoff will block the get request without one
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'}
    result = requests.get(url, headers=headers)

    # Parse the data
    src = result.content
    soup = BeautifulSoup(src, 'lxml')

    goalie_boxes = soup.find_all('div', {'class':'starting-goalies-card stat-card'})

    # find the goalie box that contains the games we are looking for
    for count, box in enumerate(goalie_boxes):
        if home_team and away_team in box.text:
            goalie_box = goalie_boxes[count]
        else:
            continue
    # retrieve the h4 headings which contain the starting goalies

    h4 = goalie_box.find_all('h4')

    # Away goalie is at element 1 and home goalie is at element 2
    away_goalie = h4[1].text
    home_goalie = h4[2].text

    return home_goalie, away_goalie
#%% converts a players name to an id number
def convert_player_to_id(team_name: str, player_name: str):
    """
    converts a player name to id
    ...
    Parameters
    ----------
    team_name: str
        abbreviation for the players team
    player_name: str
        player name string. first and last name (ex. 'Olli Jokinen')
    Returns
    -------
    player_id: int
        player id
    """
    url = f'https://statsapi.web.nhl.com/api/v1/teams'
    resp = requests.get(url)
    json_data = json.loads(resp.text)

    for team in json_data['teams']:
        if team['abbreviation'] == team_name:
            team_id = team['id']
        else:
            continue
    # Use the team id to go to team page
    url = f'https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.roster'
    resp = requests.get(url)
    json_data = json.loads(resp.text)

    team_roster = json_data['teams'][0]['roster']['roster']

    for p in team_roster:
        if p['person']['fullName'] == player_name:
            return p['person']['id']
        else:
            continue
#%% get game ids to predict
def main_get_predict_game_ids(date):
    '''This function will get games for the date specified and append them to the end of our
    dataframes. These games are games that have not yet happened and predictions for outcomes
    will be made.
    -------------
    date - str date for the day you would like to predict games (eg. 2021-01-13)'''

    url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={date}"
    resp = requests.get(url)
    raw_game_schedule = json.loads(resp.text)

    predict_ids = [] # list that will hold game ids we want to predict

    if raw_game_schedule['totalGames']==0:
        print('No games to predict on this day')

    else:
        for game in raw_game_schedule['dates'][0]['games']:
            game_id = game['gamePk']
            predict_ids.append(game_id)

    return predict_ids
#%% function to scrape games info for predictions
def scrape_prediction_game_info(game_id:int, string_date:str) -> dict:
    """
    returns an NhlGame object with parameters for the game_id provided for prediction use
    refer to: https://github.com/dword4/nhlapi on how to use the NHL API
    ...
        Parameters
        ----------
        game_id: int
            game id we are retrieving data for
        Returns
        -------
        game: NhlGame
            NhlGame object with info for the game_id provided
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url = f'https://statsapi.web.nhl.com/api/v1/game/{str(game_id)}/feed/live'
    resp = session.get(url)
    json_data = json.loads(resp.text)

    # RETRIEVE INFO REQUIRED

    # retrieve date and convert to date time
    game_date: str = json_data['gameData']['datetime']['dateTime']
    game_date = dt.datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%SZ')

    # Retrieve team names
    home_team: str = json_data["liveData"]['boxscore']['teams']['home']['team']['abbreviation']
    away_team: str = json_data["liveData"]['boxscore']['teams']['away']['team']['abbreviation']

    # Retrieve starting goalie names
    home_goalie_name, away_goalie_name = get_starting_goalies(home_team, away_team, string_date)

    # Retrieve starting goalie ids
    home_goalie_id = convert_player_to_id(home_team, home_goalie_name)
    away_goalie_id = convert_player_to_id(away_team, away_goalie_name)

    game_info = {'date':game_date, 'game_id':game_id, 'home_team':home_team,
                 'away_team':away_team, 'home_team_win':None,
                 'home_goalie_id':home_goalie_id, 'away_goalie_id':away_goalie_id,
                 'home_goalie_name':home_goalie_name,
                 'away_goalie_name':away_goalie_name}
    return game_info
#%% function to scrape teams info for predictions
def scrape_prediction_team_stats(game_id: int, string_date:str) -> List[dict]:
    """
        returns two entries in a List. The first entry is for the home team and the second is the away team.
        Each entry represents 1 game that will be played
        Refer to: https://github.com/dword4/nhlapi on how to use the NHL API
        ...
        Parameters
        ----------
        game_id: int
            game id we are retrieving data for
        Returns
        -------
        teams: List[NhlTeam]
            list containing an entry for the home team and away team playing in the same game
    """

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url = f'https://statsapi.web.nhl.com/api/v1/game/{str(game_id)}/feed/live'
    resp = session.get(url)
    json_data = json.loads(resp.text)

    # RETRIEVE STATS REQUIRED

    # retrieve date and convert to date time
    game_date: str = json_data['gameData']['datetime']['dateTime']
    game_date = dt.datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%SZ')

    # Retrieve team names
    home_team: str = json_data["liveData"]['boxscore']['teams']['home']['team']['abbreviation']
    away_team: str = json_data["liveData"]['boxscore']['teams']['away']['team']['abbreviation']

    # Retrieve starting goalie names
    home_goalie_name, away_goalie_name = get_starting_goalies(home_team, away_team, string_date)

    # Retrieve starting goalie ids
    home_goalie_id = convert_player_to_id(home_team, home_goalie_name)
    away_goalie_id = convert_player_to_id(away_team, away_goalie_name)

    # create NhlTeam objects for the home and away team
    home_team_stats = {'date':game_date, 'game_id':game_id,
                       'team':home_team, 'is_home_team':True,
                       'home_team_win':None,'goals':None,
                       'pim':None, 'shots':None,
                       'powerPlayPercentage':None,
                       'powerPlayGoals':None,
                       'powerPlayOpportunities':None,
                       'faceOffWinPercentage':None, 'blocked':None,
                       'takeaways':None, 'giveaways':None,
                       'hits':None, 'goalie_id':home_goalie_id,
                       'goalie_name':home_goalie_name}

    away_team_stats = {'date':game_date, 'game_id':game_id,
                       'team':away_team, 'is_home_team':False,
                       'home_team_win':None, 'goals':None,
                       'pim':None, 'shots':None,
                       'powerPlayPercentage':None,
                       'powerPlayGoals':None,
                       'powerPlayOpportunities':None,
                       'faceOffWinPercentage':None,'blocked':None,
                       'takeaways':None, 'giveaways':None,
                       'hits':None, 'goalie_id':away_goalie_id,
                       'goalie_name':away_goalie_name}

    teams = [home_team_stats, away_team_stats]

    return teams
#%% function to scrape goalie info for predictions
def scrape_prediction_goalie_stats(game_id: int, string_date:str) -> List[dict]:
    """
        retrieves a list of NhlGoalie containing goalie stats for all goalies that played in the game
        specified by game_id
        Refer to: https://github.com/dword4/nhlapi on how to use the NHL API
        ...
        Parameters
        ----------
        game_id: int
            game id we are retrieving data for
        Returns
        -------
        team_stats: List[NhlTeam]
            list containing an entry for the home team and away team playing in the same game
        """
    # backoff strategy to avoid maxretry errors
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url = f'https://statsapi.web.nhl.com/api/v1/game/{str(game_id)}/feed/live'
    resp = session.get(url)
    json_data = json.loads(resp.text)

    # RETRIEVE STATS REQUIRED

    # get date
    game_date = json_data['gameData']['datetime']['dateTime']
    game_date = dt.datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%SZ')

    # Get goalie team
    home_goalie_team = json_data['gameData']['teams']['home']['abbreviation']
    away_goalie_team = json_data['gameData']['teams']['away']['abbreviation']

    # Retrieve starting goalie names
    home_goalie_name, away_goalie_name = get_starting_goalies(home_goalie_team, away_goalie_team, string_date)

    # Retrieve starting goalie ids
    home_goalie_id = convert_player_to_id(home_goalie_team, home_goalie_name)
    away_goalie_id = convert_player_to_id(away_goalie_team, away_goalie_name)

    goalie_stats = {'date':game_date, 'game_id':game_id,
                    'team':home_goalie_team, 'is_home_team':True,
                    'goalie_name':home_goalie_name, 'goalie_id':home_goalie_id,
                    'timeOnIce':None, 'assists':None, 'goals':None, 'pim':None,
                    'shots':None, 'saves':None, 'powerPlaySaves':None,
                    'shortHandedSaves':None, 'evenSaves':None,
                    'shortHandedShotsAgainst':None, 'evenShotsAgainst':None,
                    'powerPlayShotsAgainst':None, 'decision':None,
                    'savePercentage':None, 'evenStrengthSavePercentage':None}
    home_goalies = []
    home_goalies.append(goalie_stats)

    # away goalies
    goalie_stats = {'date':game_date, 'game_id':game_id,
                    'team':away_goalie_team, 'is_home_team':False,
                    'goalie_name':away_goalie_name, 'goalie_id':away_goalie_id,
                    'timeOnIce':None, 'assists':None, 'goals':None, 'pim':None,
                    'shots':None, 'saves':None, 'powerPlaySaves':None,
                    'shortHandedSaves':None, 'evenSaves':None,
                    'shortHandedShotsAgainst':None, 'evenShotsAgainst':None,
                    'powerPlayShotsAgainst':None, 'decision':None,
                    'savePercentage':None, 'evenStrengthSavePercentage':None}
    away_goalies = []
    away_goalies.append(goalie_stats)

    # Merge the two lists
    goalie_stats = away_goalies + home_goalies

    return goalie_stats
#%% function to retrieve teams info for games to predict
def pull_predict_team_stats(game_ids: List[int], string_date:str) -> List[dict]:
    """
    pulls all team stats for the provided game ids
    ...
    Parameters
    ----------
    game_ids: List[int]
        list of game ids to pull team stats for
    Returns
    -------
    team_stats: List[nhl_scraper.NhlTeam]
        list of NhlTeam objects
    """

    # retrieve game by game stats for every game in the game_ids list
    team_stats = []

    for i in game_ids:
        stats_i = scrape_prediction_team_stats(i,string_date)
        team_stats += stats_i

        if len(team_stats) % 500 == 0:  # Progress bar
            print(str(0.5 * len(team_stats) / len(game_ids) * 100) + ' percent done retrieving game data/stats.')

    return team_stats
#%% function retrieve goalies info for games to predict
def pull_predict_goalie_stats(game_ids: List[int], string_date:str) -> List[dict]:
    """
        pulls all goalie stats for the provided game ids
        ...
        Parameters
        ----------
        game_ids: List[int]
            list of game ids to pull team stats for
        Returns
        -------
        goalie_stats: List[nhl_scraper.NhlGoalie]
            list of NhlGoalie objects
        """

    goalie_stats=[]
    for i in game_ids:
        goalies_i = scrape_prediction_goalie_stats(i, string_date)
        goalie_stats += goalies_i

        if len(goalie_stats) % 250 == 0:  # Progress bar # todo fix progress bar to account for more goalies than game ids
            print(str(0.5 * len(goalie_stats) / len(game_ids) * 100) + ' percent done retrieving goalie data.')

    return goalie_stats
#%%function to retrieve games info for games to predict
def pull_predict_game_info(game_ids: List[int], string_date:str) -> List[dict]:
    """
    pulls all game_info for the provided game ids
    ...
    Parameters
    ----------
    game_ids: List[int]
        list of game ids to pull team stats for
    string_date: str
        date for which we are predicting game 01-25-2021
    Returns
    -------
    games_info: List[nhl_scraper.NhlGame]
        list of NhlGame objects
    """

    # retrieve game by game info for every game in the game_ids list
    games_info = []

    for i in game_ids:
        game_i = scrape_prediction_game_info(i, string_date)
        games_info.append(game_i)

        if len(games_info) % 500 == 0:  # Progress bar
            print(str(len(games_info) / len(game_ids) * 100) + ' percent done retrieving game data/stats.')
    return games_info
#%% make teams dataframe
def make_teams_df(team_stats: List[dict]) -> pd.DataFrame:
    """
        makes a dataframe from a list of NhlTeam objects
        ...
        Parameters
        ----------
        team_stats: List[nhl_scraper.NhlTeam]
            list of NhlTeam objects
        Returns
        -------
        teams_df: pd.DataFrame
            each row of dataframe represents stats for 1 team in 1 game. Therefore each game will have
            2 rows one for the home team and one for away.
        """

    teams_df = pd.DataFrame(team_stats)
    return teams_df
#%% make goalies dataframe
def make_goalies_df(goalie_stats: List[dict]) -> pd.DataFrame:
    """
        makes a dataframe from a list of NhlGoalie objects
        ...
        Parameters
        ----------
        goalie_stats: List[nhl_scraper.NhlGoalie]
            list of NhlGoalie objects
        Returns
        -------
        goalies_df: pd.DataFrame
            each row of dataframe represents stats for 1 goalie in 1 game. Therefore each game will have
            at least 2 rows.
        """

    goalies_df = pd.DataFrame(goalie_stats)
    return goalies_df
#%% make games datatframe
def make_games_df(games_info: List[dict]) -> pd.DataFrame:
    """
        main dataframe that will eventually get fed to the machine learning model
        ...
        Parameters
        ----------
        team_stats: List[nhl_scraper.NhlGoalie]
            list of NhlTeam objects
        Returns
        -------
        games_df: pd.DataFrame
            each row of dataframe represents 1 NHL game
        """
    games_df = pd.DataFrame(games_info)
    return games_df
#%% convert stat categories to numerical
def convert_numerical(teams_df: pd.DataFrame, goalies_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
        convert non-numerical features in dataframes to numerical
        ...
        Parameters
        ----------
        teams_df: pd.DataFrame
            dataframe containing team stats
        goalies_df: pd.DataFrame
            dataframe containing goalies stats
        Returns
        -------
        teams_numerical_df: pd.DataFrame
            teams_df converted to numerical
        goalies_numerical_df: pd.DataFrame
            goalies_numerical_df: pd.DataFrame
        """
    # CONVERT OBJECTS TO NUMERICAL
    teams_numerical_df = teams_df.copy()
    goalies_numerical_df = goalies_df.copy()
    # powerPlayPercentage
    teams_numerical_df['powerPlayPercentage'] = teams_numerical_df['powerPlayPercentage'].astype(float)
    # faceOffWinPercentage
    teams_numerical_df['faceOffWinPercentage'] = teams_numerical_df['faceOffWinPercentage'].astype(float)
    # Convert Goalie timeOnIce to Minutes
    goalies_numerical_df['timeOnIce'] = goalies_numerical_df['timeOnIce'].map(convert_minutes)

    # Reset index
    teams_numerical_df.reset_index(inplace=True)
    goalies_numerical_df.reset_index(inplace=True)

    return teams_numerical_df, goalies_numerical_df
#%% add pdos to dataframe
def add_pdo(teams_df: pd.DataFrame, goalies_df: pd.DataFrame) -> pd.DataFrame:
    """
        adds pdo as a stat to the teams_df. will also add evenStrengthGoals, evenStrengthShootingPercent
        and evenStrengthShots.
        ...
        Parameters
        ----------
        teams_df: pd.DataFrame
            dataframe containing team stats
        goalies_df: pd.DataFrame
            dataframe containing goalies stats
        Returns
        -------
        teams_df: pd.DataFrame
            teams_df with added stats
        """
    # get game ids
    game_ids = teams_df['game_id'].to_list()
    game_ids = remove_duplicates(game_ids)

    pdos = []

    for id in game_ids:
        goalies_filtered_df = goalies_df[goalies_df['game_id'] == id]

        # Filter to the home team goalies that played in that game
        home_goalies_filtered_df = goalies_filtered_df[goalies_filtered_df['is_home_team'] == True]

        # Filter to the away team goalies that played in that game
        away_goalies_filtered_df = goalies_filtered_df[goalies_filtered_df['is_home_team'] == False]

        # Away shots are taken from the home goalie stats and vice versa
        away_es_shots = home_goalies_filtered_df['evenShotsAgainst'].sum()
        home_es_shots = away_goalies_filtered_df['evenShotsAgainst'].sum()

        away_es_goals = home_goalies_filtered_df['evenShotsAgainst'].sum() - home_goalies_filtered_df['evenSaves'].sum()
        home_es_goals = away_goalies_filtered_df['evenShotsAgainst'].sum() - away_goalies_filtered_df['evenSaves'].sum()

        # Calculate ES Sh%
        home_es_sh_percent = home_es_goals / home_es_shots
        away_es_sh_percent = away_es_goals / away_es_shots

        # Calculate ES Sv%
        home_es_sv_percent = (away_es_shots - away_es_goals) / away_es_shots
        away_es_sv_percent = (home_es_shots - home_es_goals) / home_es_shots

        # Calculate PDO
        home_PDO = home_es_sh_percent + home_es_sv_percent
        away_PDO = away_es_sh_percent + away_es_sv_percent

        # Create dictionary 1 entry for each team
        pdo_dict = [{'game_id': id, 'pdo': home_PDO, 'evenStrengthGoals' : home_es_goals, 'evenStrengthShots' : home_es_shots, 'evenStrengthShootingPercent' : home_es_sh_percent, 'is_home_team' : True},
                    {'game_id': id, 'pdo': away_PDO, 'evenStrengthGoals' : away_es_goals, 'evenStrengthShots' : away_es_shots, 'evenStrengthShootingPercent' : away_es_sh_percent, 'is_home_team' : False}]
        # Append to list
        pdos += pdo_dict

    # Create PDO dataframe
    pdo_df = pd.DataFrame(pdos)

    # Merge PDO's into teams_df
    teams_df = pd.merge(teams_df, pdo_df, left_on=['game_id', 'is_home_team'],
                          right_on=['game_id', 'is_home_team'], how='left')

    return teams_df
#%% add shooting percentage to dataframe
def add_sh_per(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    adds shooting percentage as a stat to the teams_df
    ...
    Parameters
    ----------
    teams_df: pd.DataFrame
        dataframe containing team stats
    Returns
    -------
    teams_df: pd.DataFrame
        teams_df with shooting percentage added
    """
    teams_df['Shooting_Percent'] = teams_df['goals']/teams_df['shots']

    return teams_df
#%% calculate rolling stats
def add_rolling(period, df, stat_columns, is_goalie=False):
    """
    creates rolling average stats in in dataframe provided
    ...
    Parameters
    ----------
    period: int
        the period for which we want to create rolling average
    df: pd.DataFrame
        dataframe to process
    stat_columns: List['str']
        list of columns in dataframe to create rolling stats for
    Returns
    -------
    df: pd.DataFrame
        dataframe with rolling stats added
    """
    for s in stat_columns:
        if 'object' in str(df[s].dtype): continue
        df[s+'_'+str(period)+'_avg'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).mean())
        df[s+'_'+str(period)+'_std'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).std())
        df[s+'_'+str(period)+'_skew'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).skew())

    return df
#%% calculate differential stats
def get_diff_df(df, name, is_goalie=False):
    """
    calculated stat differentials between home and away team
    ...
    Parameters
    ----------
    df: pd.DataFrame
        dataframe to process
    is_goalie: bool
        if this is a goalie dataframe stats will be grouped by goalies instead of team
    Returns
    -------
    diff_df: pd.DataFrame
        dataframe with calculated stat differentials
    """
    # Sort by date
    df = df.sort_values(by='date').copy()
    newindex = df.groupby('date')['date'].apply(lambda x: x + np.arange(x.size).astype(np.timedelta64))
    df = df.set_index(newindex).sort_index()

    # get stat columns
    stat_cols = [x for x in df.columns if 'int' in str(df[x].dtype)]
    stat_cols.extend([x for x in df.columns if 'float' in str(df[x].dtype)])

    #add rolling stats to the data frame
    df = add_rolling(3, df, stat_cols)
    df = add_rolling(7, df, stat_cols)
    df = add_rolling(14, df, stat_cols)
    df = add_rolling(41, df, stat_cols)
    df = add_rolling(82, df, stat_cols)

    # reset stat columns to just the sma features (removing the original stats)
    df.drop(columns=stat_cols, inplace=True)
    stat_cols = [x for x in df.columns if 'int' in str(df[x].dtype)]
    stat_cols.extend([x for x in df.columns if 'float' in str(df[x].dtype)])

    # shift results so that each row is a pregame stat
    df = df.reset_index(drop=True)
    df = df.sort_values(by='date')

    for s in stat_cols:
        if is_goalie:
            df[s] = df.groupby('goalie_id')[s].shift(1)
        else:
            df[s] = df.groupby('team')[s].shift(1)

    # calculate differences in pregame stats from home vs. away teams
    away_df = df[~df['is_home_team']].copy()
    away_df = away_df.set_index('game_id')
    away_df = away_df[stat_cols]

    home_df = df[df['is_home_team']].copy()
    home_df = home_df.set_index('game_id')
    home_df = home_df[stat_cols]

    diff_df = home_df.subtract(away_df, fill_value=0)
    diff_df = diff_df.reset_index()

    # clean column names
    for s in stat_cols:
        diff_df[name + "_" + s] = diff_df[s]
        diff_df.drop(columns=s, inplace=True)

    return diff_df
#%% add goalie rest
def goalie_rest(goalies_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates how many rest days a goalie has had with a maximum value of 30 days
    ...
    Parameters
    ----------
    goalies_df: pd.DataFrame
        goalies dataframe
    games_df: pd.Dataframe
        games dataframe
    Returns
    -------
    games_df: pd.DataFrame
        dataframe with goalie rest added
    """
    # It's easier with the way the goalie df is setup to calculate this in here than merge into the main dataframe
    goalies_df['goalie_rest'] = goalies_df.groupby('goalie_id')['date'].diff().dt.days

    # The first teams games in the DF are NaN as there are no previous reference points.
    # We will fill these in with the max value of 30 days as these games were at the start of the season
    goalies_df['goalie_rest'].fillna(30, inplace=True)

    # If the days rest is over 30 just make it 30
    goalies_df.loc[goalies_df["goalie_rest"] > 30, "goalie_rest"] = 30

    # Make a dataframe just containing goalie rest data
    goalie_rest = goalies_df[['game_id', 'goalie_id', 'goalie_rest']]
    # Rename to Home and Away Goalie Rest
    home_goalie_rest = goalie_rest.rename({'goalie_rest': 'home_goalie_rest'}, axis=1)
    away_goalie_rest = goalie_rest.rename({'goalie_rest': 'away_goalie_rest'}, axis=1)

    # Merge into main dataframe
    games_df = pd.merge(games_df, home_goalie_rest, left_on=['game_id', 'home_goalie_id'],
                            right_on=['game_id', 'goalie_id'], how='left')

    games_df = pd.merge(games_df, away_goalie_rest, left_on=['game_id', 'away_goalie_id'],
                            right_on=['game_id', 'goalie_id'], how='left')

    # Remove some columns
    games_df.drop(['goalie_id_x','goalie_id_y'], axis=1, inplace=True)

    return games_df
#%% add team rest
def team_rest(goalies_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates how many rest days a teams has had with a maximum value of 7 days
    ...
    Parameters
    ----------
    goalies_df: pd.DataFrame
        goalies dataframe
    games_df: pd.Dataframe
        games dataframe
    Returns
    -------
    games_df: pd.DataFrame
        dataframe with team rest added
    """
    # TEAM DAYS REST
    # It's easier with the way the goalie df is setup to calculate this in here than merge into the main dataframe
    # Convert date to datetime in goalie dataframe
    goalies_df['team_rest'] = goalies_df.groupby('team')['date'].diff().dt.days
    # The first teams games in the DF are NaN as there are no previous reference points.
    # We will fill these in with the max value of 7 days as these games were at the start of the 2010 season
    goalies_df['team_rest'].fillna(7, inplace=True)
    # If the days rest is over 7 just make it 7
    goalies_df.loc[goalies_df["team_rest"] > 7, "team_rest"] = 7
    # Make a dataframe just containing team rest data
    team_rest = goalies_df[['game_id', 'goalie_id', 'team_rest']]

    # Rename to Home/Away Team Rest
    home_team_rest = team_rest.rename({'team_rest': 'home_team_rest'}, axis=1)
    away_team_rest = team_rest.rename({'team_rest': 'away_team_rest'}, axis=1)

    # Convert data to same types
    home_team_rest['game_id'] = home_team_rest['game_id'].astype('string')
    home_team_rest['goalie_id'] = home_team_rest['goalie_id'].astype('string')
    away_team_rest['game_id'] = away_team_rest['game_id'].astype('string')
    away_team_rest['goalie_id'] = away_team_rest['goalie_id'].astype('string')

    # Merge into main dataframe
    games_df = pd.merge(games_df, home_team_rest, left_on=['game_id', 'home_goalie_id'],
                            right_on=['game_id', 'goalie_id'], how='left')

    games_df = pd.merge(games_df, away_team_rest, left_on=['game_id', 'away_goalie_id'],
                            right_on=['game_id', 'goalie_id'], how='left')

    # Drop some columns
    games_df.drop(['goalie_id_x', 'goalie_id_y'], axis=1, inplace=True)

    return games_df
#%% add rolling win percentages to data frame
def rolling_win_percentage(games_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    creates a moving average win percentage
    ...
    Parameters
    ----------
    games_df: pd.DataFrame
        games dataframe
    period: int
        period for which we want to calculate win percentage
    Returns
    -------
    games_df: pd.DataFrame
        dataframe with rolling win percentage added
    """
    # Target Encoding, this will create a period SMA win percentage columns for the home and away teams
    column_names = ['home_win_percent_'+str(period)+'_avg','away_win_percent_'+str(period)+'_avg']

    for x in column_names:
        if x == 'home_win_percent_'+str(period)+'_avg':
            games_df[x] = games_df.groupby('home_team')['home_team_win'].apply(lambda x: x.rolling(period).mean()).shift(1)
        else:
            games_df[x] = games_df.groupby('away_team')['home_team_win'].apply(lambda x: x.rolling(period).mean()).shift(1)
    return games_df
#%% make predictions
def make_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    takes the prediction dataframe and runs XGBoost model to predict games
    ...
    Parameters
    ----------
    prediction_df: pd.DataFrame
        prediction dataframe
    Returns
    -------
    predict_df: pd.DataFrame
        prediction_df with added win percentage from XGBoost
    """
    # load model
    with open('/Users/patrickpetanca/projects/nhl_model/data/xgb_model_opt.pkl', 'rb') as f:
        model = pickle.load(f)

    # Create prediction set
    X = prediction_df.drop(columns=['game_id', 'home_team', 'away_team', 'date', 'home_goalie_id', 'away_goalie_id', 'home_team_win'])

    # Reorder columns so training and prediction sets match
    cols_when_model_builds = model.get_booster().feature_names
    X = X[cols_when_model_builds]

    # Create prediction dataframe
    predict_df = prediction_df[['game_id','home_team','away_team','date','home_goalie_id','away_goalie_id','home_goalie_name','away_goalie_name']]

    predict_df['xgb_home_win'] = model.predict(X).astype('bool')
    predict_df['xgb_home_win_percent'] = model.predict_proba(X)[:,1]

    return(predict_df)
#%% fast elo
def fast_elo_ratings(df):
    """
    creates fast moving elo ratings
    ...
    Parameters
    ----------
    df: pd.DataFrame
        games dataframe
    Returns
    -------
    df: pd.DataFrame
        with fast moving elo ratings added
    """
    ratings = {}
    for x in df.home_team.unique():
        ratings[x] = EloCompetitor()
    for x in df.away_team.unique():
        ratings[x] = EloCompetitor()

    home_team_elo = []
    away_team_elo = []
    elo_exp = []

    df = df.sort_values(by='date').reset_index(drop=True)
    for i, r in df.iterrows():
        # get pre-game ratings
        elo_exp.append(ratings[r.home_team].expected_score(ratings[r.away_team]))
        home_team_elo.append(ratings[r.home_team].rating)
        away_team_elo.append(ratings[r.away_team].rating)

        # update ratings
        if r.home_team_win:
            ratings[r.home_team].beat(ratings[r.away_team])
        else:
            ratings[r.away_team].beat(ratings[r.home_team])

    df['elo_exp'] = elo_exp
    df['home_team_elo'] = home_team_elo
    df['away_team_elo'] = away_team_elo

    return df
#%% slow elo
def slow_elo_ratings(df):
    """
    creates slow moving elo ratings
    ...
    Parameters
    ----------
    df: pd.DataFrame
        games dataframe
    Returns
    -------
    df: pd.DataFrame
        with slow moving elo ratings added
    """
    ratings = {}

    # Obtain team names
    for x in df.home_team.unique():
        ratings[x]=EloCompetitor()
        ratings[x]._k_factor = 16
    for x in df.away_team.unique():
        ratings[x]=EloCompetitor()
        ratings[x]._k_factor = 16

    home_team_elo = []
    away_team_elo = []
    elo_exp = []

    df = df.sort_values(by='date').reset_index(drop=True)
    for i, r in df.iterrows():
        # get pre-game ratings
        elo_exp.append(ratings[r.home_team].expected_score(ratings[r.away_team]))
        home_team_elo.append(ratings[r.home_team].rating)
        away_team_elo.append(ratings[r.away_team].rating)

        # update ratings
        if r.home_team_win:
            ratings[r.home_team].beat(ratings[r.away_team])
        else:
            ratings[r.away_team].beat(ratings[r.home_team])

    df['slow_elo_exp'] = elo_exp
    df['home_team_slow_elo'] = home_team_elo
    df['away_team_slow_elo'] = away_team_elo

    return df
#%% glicko
def glicko(df):
    """
    creates glicko ratings
    ...
    Parameters
    ----------
    df: pd.DataFrame
        games dataframe
    Returns
    -------
    df: pd.DataFrame
        with fast moving glicko added
    """
    ratings = {}
    for x in df.home_team.unique():
        ratings[x] = GlickoCompetitor()
    for x in df.away_team.unique():
        ratings[x] = GlickoCompetitor()

    home_team_glick = []
    away_team_glick = []
    glick_exp = []

    df = df.sort_values(by='date').reset_index(drop=True)
    for i, r in df.iterrows():
        # get pregame ratings
        glick_exp.append(ratings[r.home_team].expected_score(ratings[r.away_team]))
        home_team_glick.append(ratings[r.home_team].rating)
        away_team_glick.append(ratings[r.away_team].rating)
        # update ratings
        if r.home_team_win:
            ratings[r.home_team].beat(ratings[r.away_team])
        else:
            ratings[r.away_team].beat(ratings[r.home_team])

    df['glick_exp'] = glick_exp
    df['home_team_glick'] = home_team_glick
    df['away_team_glick'] = away_team_glick

    return df
#%% trueskill
def trueskill(df):
    """
    creates trueskill ratings
    ...
    Parameters
    ----------
    df: pd.DataFrame
        games dataframe
    Returns
    -------
    df: pd.DataFrame
        with trueskill ratings added
    """
    ratings = {}
    for x in df.home_team.unique():
        ratings[x] = Rating(25)
    for x in df.away_team.unique():
        ratings[x] = Rating(25)
    for x in df.home_goalie_id.unique():
        ratings[x] = Rating(25)
    for x in df.away_goalie_id.unique():
        ratings[x] = Rating(25)

    ts_quality = []
    goalie_ts_diff = []
    team_ts_diff = []
    home_goalie_ts = []
    away_goalie_ts = []
    home_team_ts = []
    away_team_ts = []
    df = df.sort_values(by='date').copy()

    for i, r in df.iterrows():
        # get pre-match trueskill ratings from dict
        match = [(ratings[r.home_team], ratings[r.home_goalie_id]),
                 (ratings[r.away_team], ratings[r.away_goalie_id])]
        ts_quality.append(quality(match))
        goalie_ts_diff.append(ratings[r.home_goalie_id].mu - ratings[r.away_goalie_id].mu)
        team_ts_diff.append(ratings[r.home_team].mu - ratings[r.away_team].mu)
        home_goalie_ts.append(ratings[r.home_goalie_id].mu)
        away_goalie_ts.append(ratings[r.away_goalie_id].mu)
        home_team_ts.append(ratings[r.home_team].mu)
        away_team_ts.append(ratings[r.away_team].mu)

        if r.date < df.date.max():
            # update ratings dictionary with post-match ratings
            if r.home_team_win:
                match = [(ratings[r.home_team], ratings[r.home_goalie_id]),
                         (ratings[r.away_team], ratings[r.away_goalie_id])]
                [(ratings[r.home_team], ratings[r.home_goalie_id]),
                 (ratings[r.away_team], ratings[r.away_goalie_id])] = rate(match)
            else:
                match = [(ratings[r.away_team], ratings[r.away_goalie_id]),
                         (ratings[r.home_team], ratings[r.home_goalie_id])]
                [(ratings[r.away_team], ratings[r.away_goalie_id]),
                 (ratings[r.home_team], ratings[r.home_goalie_id])] = rate(match)

    df['ts_game_quality'] = ts_quality
    df['goalie_ts_diff'] = goalie_ts_diff
    df['team_ts_diff'] = team_ts_diff
    df['home_goalie_ts'] = home_goalie_ts
    df['away_goalie_ts'] = away_goalie_ts
    df['home_team_ts'] = home_team_ts
    df['away_team_ts'] = away_team_ts

    return df
#%% get game ids to predict input
predict_ids = main_get_predict_game_ids('2021-04-22')
#%% string date input
string_date = '04-22-2021'
#%% remove any postponed games
#predict_ids.remove(2020020191)
#%% retrieve game by game information for all predict game ids pulled
predict_games_info = pull_predict_game_info(predict_ids, string_date)
#%% retrieve team game by game stats for all predict game ids pulled
predict_team_stats = pull_predict_team_stats(predict_ids, string_date)
#%% retrieve goalie game by game stats for all predict game ids pulled
predict_goalie_stats = pull_predict_goalie_stats(predict_ids, string_date)
#%% import pickle files
with open('/Users/patrickpetanca/projects/nhl_model/data/team_stats.pkl', 'rb') as f:
    team_stats = pickle.load(f)
    
with open('/Users/patrickpetanca/projects/nhl_model/data/goalie_stats.pkl', 'rb') as f:
    goalie_stats = pickle.load(f)
    
with open('/Users/patrickpetanca/projects/nhl_model/data/games_info.pkl', 'rb') as f:
    games_info = pickle.load(f)
#%% append prediction games to end of information lists
team_stats = team_stats + predict_team_stats
goalie_stats = goalie_stats + predict_goalie_stats
games_info = games_info + predict_games_info
#%% make dataframes
teams_df = make_teams_df(team_stats)
goalies_df = make_goalies_df(goalie_stats)
games_df = make_games_df(games_info)
#%% If it cannot find a goalie id replace the id with 0
games_df['home_goalie_id'].fillna(value=0, inplace=True)
games_df['away_goalie_id'].fillna(value=0, inplace=True)
games_df['home_goalie_id'] = games_df['home_goalie_id'].astype(int)
games_df['away_goalie_id'] = games_df['away_goalie_id'].astype(int)

goalies_df['goalie_id'].fillna(value=0, inplace=True)
goalies_df['goalie_id'] = goalies_df['goalie_id'].astype(int)

teams_df['goalie_id'].fillna(value=0, inplace=True)
teams_df['goalie_id'] = teams_df['goalie_id'].astype(int)
#%% convert to numerical
teams_df, goalies_df = convert_numerical(teams_df, goalies_df)
#%% add pdo
teams_df = add_pdo(teams_df, goalies_df)
#%% add shooting percent
teams_df = add_sh_per(teams_df)
#%% remove columns that will not be used in teams_df and goalies_df
teams_df.drop(['index'], axis=1, inplace=True)
goalies_df.drop(['index', 'assists', 'goals', 'pim', 'decision'], axis=1, inplace=True)
#%% convert ids to strings
teams_df['game_id'] = teams_df['game_id'].map(str)
teams_df['goalie_id'] = teams_df['goalie_id'].map(str)
goalies_df['game_id'] = goalies_df['game_id'].map(str)
goalies_df['goalie_id'] = goalies_df['goalie_id'].map(str)
games_df['game_id'] = games_df['game_id'].map(str)
games_df['home_goalie_id'] = games_df['home_goalie_id'].map(str)
games_df['away_goalie_id'] = games_df['away_goalie_id'].map(str)
#%% reset indexes
games_df = games_df.reset_index(drop=True)
players_df = teams_df.reset_index(drop=True)
goalies_df = goalies_df.reset_index(drop=True)
#%% create rolling player stats in main games dataframe

games_df = pd.merge(left=games_df, right=get_diff_df(teams_df, 'teams'),
              on='game_id', how='left')

print(games_df.shape)
#%% create rolling goalie stats in main games dataframe
games_df = pd.merge(left=games_df, right=get_diff_df(goalies_df, 'goalies', is_goalie=True),
              on='game_id', how='left')
#%% drop duplicates due to multiple goalies playing in one game
# todo confirm if the first or last game should be kept
games_df.drop_duplicates(subset=['game_id'], keep="last", inplace=True)

print(games_df.shape)
#%% add goalie rest
games_df = goalie_rest(goalies_df, games_df)
#%% add team rest
games_df = team_rest(goalies_df, games_df)
#%% add power rankings
games_df = fast_elo_ratings(games_df)
games_df = slow_elo_ratings(games_df)
games_df = glicko(games_df)
games_df = trueskill(games_df)
#%% rolling win percentage
# first remove the games that have not yet happened
rolling_win_df = games_df.drop(games_df.tail(len(predict_ids)).index)
rolling_win_toappend_df = games_df.tail(len(predict_ids))

days = [5,10,20,41,82]

for d in days:
    # Target Encoding, this will create a period SMA win percentage columns for the home and away teams
    encode_me = ['home_win_percent_' + str(d) + '_avg', 'away_win_percent_' + str(d) + '_avg']

    for x in encode_me:
        if x == 'home_win_percent_' + str(d) + '_avg':
            rolling_win_df[x] = rolling_win_df.groupby('home_team')['home_team_win'].apply(lambda x: x.rolling(d).mean())
        else:
            rolling_win_df[x] = rolling_win_df.groupby('away_team')['home_team_win'].apply(lambda x: x.rolling(d).mean())
#%% Append games to predict back to bottom of dataframe so we can shift the win percentage
# results down into the games to predict
rolling_win_df = rolling_win_df.append(rolling_win_toappend_df, ignore_index=True)
rolling_win_df.reset_index(inplace=True)

for d in days:
    # Target Encoding, this will create a period SMA win percentage columns for the home and away teams
    encode_me = ['home_win_percent_' + str(d) + '_avg', 'away_win_percent_' + str(d) + '_avg']

    for x in encode_me:
        if x == 'home_win_percent_' + str(d) + '_avg':
            rolling_win_df[x] = rolling_win_df.groupby('home_team')[x].shift(1)
        else:
            rolling_win_df[x] = rolling_win_df.groupby('away_team')[x].shift(1)

games_df = rolling_win_df
#%% Retrieve last rows which are just the games we are predicting
prediction_df = games_df.tail(len(predict_ids))

prediction_df.reset_index(inplace=True, drop=True)
#%% make predictions
predictions = make_predictions(prediction_df)

todays_date = str(dt.datetime.today()).split()[0]