#%%
'''
functions to scrape nhl.com api for data
'''
import datetime as dt
import json
from typing import List, Dict
from bs4 import BeautifulSoup
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
#%%
# get game ids from nhl.com API
def get_game_ids (season: int) -> List[int]:
    '''
    Retrieves all of the game ids for the provided season

    Arguments:
        season (int): the season for which you want to retrieve game ids (ex: 20192020)

    Returns:
        List[int]: a list containing all regular season game ids for that season
    '''

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    season_str: str = str(season)
    url: str = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season_str}&gameType=R"
    resp = session.get(url)
    raw_schedule = json.loads(resp.text)
    schedule = raw_schedule['dates']
    # each entry in schedule is a day in the NHL. Each 'games' key contains games on that day
    # Therefore we need a nested loop to retrieve all games

    game_ids=[]

    for day in schedule:
        # Retrieve list that shows all games played on that day
        games = day['games']
        # Loop through games and retrieve ids
        for game in games:
            game_id = game['gamePk']
            game_ids.append(game_id)
    return game_ids
#%% scrape team stats
def scrape_team_stats(game_id: int) -> List[Dict]:
    """
        returns two entries in a List.
        The first entry is stats for the home team and the second is stats for the away team.
        Each entry represents 1 game played.
        Refer to: https://github.com/dword4/nhlapi on how to use the NHL API

        Arguments
            game_id (int): game id we are retrieving data for

        Returns
            List[dict]: list containing an entry for the home team and away team playing in the
                        same game
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

    # collect list of teamSkaterStats we want to retrieve from json data
    team_skater_stats_home = json_data["liveData"]\
        ["boxscore"]['teams']['home']['teamStats']['teamSkaterStats']
    team_skater_stats_away = json_data["liveData"]\
        ["boxscore"]['teams']['away']['teamStats']['teamSkaterStats']

    # Starting goalies
    # spot checked a few APIs and it seems like the starting goalie will be listed last in the json
    # file if he was pulled. The goalie that finishes the game will be listed first (0).
    home_team_starting_goalie_id = json_data["liveData"]\
        ['boxscore']['teams']['home']['goalies'][-1]
    away_team_starting_goalie_id = json_data["liveData"]\
        ['boxscore']['teams']['away']['goalies'][-1]
    home_team_starting_goalie_name = json_data["liveData"]\
        ['boxscore']['teams']['home']['players']\
            ['ID'+str(home_team_starting_goalie_id)]['person']['fullName']
    away_team_starting_goalie_name = json_data["liveData"]\
        ['boxscore']['teams']['away']['players']\
            ['ID'+str(away_team_starting_goalie_id)]['person']['fullName']

    # retrieve outcome (same for both home team and away team)
    if not json_data['liveData']['linescore']['hasShootout']:
        if (json_data["liveData"]["boxscore"]\
            ['teams']['home']['teamStats']['teamSkaterStats']['goals'] >
            json_data["liveData"]["boxscore"]\
                ['teams']['away']['teamStats']['teamSkaterStats']['goals']):
            home_team_win = True
        if (json_data["liveData"]["boxscore"]\
            ['teams']['home']['teamStats']['teamSkaterStats']['goals'] <
            json_data["liveData"]["boxscore"]\
                ['teams']['away']['teamStats']['teamSkaterStats']['goals']):
            home_team_win = False
    if json_data['liveData']['linescore']['hasShootout']:
        if (json_data['liveData']['linescore']['shootoutInfo']['home']['scores'] >
            json_data['liveData']['linescore']['shootoutInfo']['away']['scores']):
            home_team_win = True
        if (json_data['liveData']['linescore']['shootoutInfo']['home']['scores'] <
            json_data['liveData']['linescore']['shootoutInfo']['away']['scores']):
            home_team_win = False

    # create dictionaries for the home and away team
    if game_id == 2020020215: # manually entering incorrect input data in NHL API for this game
        home_team_stats = {'date':game_date, 'game_id':game_id,
                           'team':home_team, 'is_home_team':True,
                           'home_team_win':False,
                           'goalie_id':home_team_starting_goalie_id,
                           'goalie_name':home_team_starting_goalie_name}

        home_team_stats.update(team_skater_stats_home)

        away_team_stats = {'date':game_date, 'game_id':game_id,
                           'team':away_team, 'is_home_team':False,
                           'home_team_win':False,
                           'goalie_id':away_team_starting_goalie_id,
                           'goalie_name':away_team_starting_goalie_name}

        away_team_stats.update(team_skater_stats_away)

    else:
        home_team_stats = {'date':game_date, 'game_id':game_id,
                           'team':home_team, 'is_home_team':True,
                           'home_team_win':home_team_win,
                           'goalie_id':home_team_starting_goalie_id,
                           'goalie_name':home_team_starting_goalie_name}

        home_team_stats.update(team_skater_stats_home)

        away_team_stats = {'date':game_date, 'game_id':game_id,
                           'team':away_team, 'is_home_team':False,
                           'home_team_win':home_team_win,
                           'goalie_id':away_team_starting_goalie_id,
                           'goalie_name':away_team_starting_goalie_name}

        away_team_stats.update(team_skater_stats_away)

    teams = [home_team_stats, away_team_stats]

    return teams
#%% scrape player stats
def scrape_player_stats(game_id: int) -> List[Dict]:
    """
    retrieves all player stats for the specified game_id
    Refer to: https://gitlab.com/dword4/nhlapi on how to use the NHL API

    Arguments
        game_id (int): Game id for which all player stats will be retrieved by

    Returns
        List[Dict]: List containing stats for all players that played in the game.
                    Each Dict represents one player

    """

    # backoff strategy to avoid max retry errors
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

    # get teams
    home_team = json_data['gameData']['teams']['home']['abbreviation']
    away_team = json_data['gameData']['teams']['away']['abbreviation']

    # get home player ids
    player_ids = list(json_data['liveData']['boxscore']['teams']['home']['players'])

    # scrape home players first
    players = []

    for i in player_ids:
        player_id = json_data['liveData']['boxscore']['teams']['home']['players'][i]['person']['id']
        player_name = json_data['liveData']['boxscore']['teams']['home']['players'][i]['person']\
            ['fullName']
        position = json_data['liveData']['boxscore']['teams']['home']['players'][i]['position']\
            ['code']
        # skater stats will have a skaterStat key
        # goalie will have goalieStat and players who didn't play will have nothing
        try:
            stats = json_data['liveData']['boxscore']['teams']['home']['players'][i]['stats']\
                ['skaterStats']
            #create dictonary for player
            player = {'date':game_date, 'game_id':game_id, 'team':home_team,
                      'is_home_team':True, 'player_name':player_name,
                      'player_id':player_id, 'position':position}
            player.update(stats)
            players.append(player)

        except KeyError:
            pass


    # scrape away players
    # get away player ids
    player_ids = list(json_data['liveData']['boxscore']['teams']['away']['players'])

    for i in player_ids:
        player_id = json_data['liveData']['boxscore']['teams']['away']['players'][i]['person']['id']
        player_name = json_data['liveData']['boxscore']['teams']['away']['players'][i]['person']\
            ['fullName']
        position = json_data['liveData']['boxscore']['teams']['away']['players'][i]['position']\
            ['code']
        # skater stats will have a skaterStat key
        # goalie will have goalieStat and players who didn't play will have nothing
        try:
            stats = json_data['liveData']['boxscore']['teams']['away']['players'][i]\
                ['stats']['skaterStats']
            #create dictonary for player
            player = {'date':game_date, 'game_id':game_id, 'team':away_team,
                      'is_home_team':False, 'player_name':player_name,
                      'player_id':player_id, 'position':position}
            player.update(stats)
            players.append(player)

        except KeyError:
            pass

    return players

#%% scrape goalie stats
def scrape_goalie_stats(game_id: int) -> List[Dict]:
    """
        retrieves a list of dictionaries containing goalie stats for all 
        goalies that played in the game specified by game_id.
        Each dictionary represents one goalie.
        Refer to: https://github.com/dword4/nhlapi on how to use the NHL API

        Arguments
            game_id (int): game id we are retrieving data for
        
        Returns
            List[Dict]: list containing an entry for the home team and away team playing in the
                        same game.
        """

    # backoff strategy to avoid max retry errors
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

    # Get goalie IDs
    home_goalie_id = json_data['liveData']['boxscore']['teams']['home']['goalies']
    away_goalie_id = json_data['liveData']['boxscore']['teams']['away']['goalies']

    # Get goalie names
    home_goalie_names = []
    away_goalie_names = []

    for i in home_goalie_id: # for loop to iterate through list of home goalies that played this game
        j = json_data['liveData']['boxscore']['teams']['home']['players']['ID' + str(i)]['person']['fullName']
        home_goalie_names.append(j)
    for i in away_goalie_id: # for loop to iterate through list of away goalies that played this game
        j = json_data['liveData']['boxscore']['teams']['away']['players']['ID' + str(i)]['person']['fullName']
        away_goalie_names.append(j)

    # Get goalie stats
    home_goalie_stats = []
    away_goalie_stats = []
    for i in home_goalie_id: # for loop to iterate through list of home goalies that played this game
        j = json_data['liveData']['boxscore']['teams']['home']['players']['ID' + str(i)]['stats']['goalieStats']
        home_goalie_stats.append(j)
    for i in away_goalie_id: # for loop to iterate through list of home goalies that played this game
        j = json_data['liveData']['boxscore']['teams']['away']['players']['ID' + str(i)]['stats']['goalieStats']
        away_goalie_stats.append(j)

    # make home goalie list. for loop needed as there could be more than 2 goalies playing in 1 game
    home_goalies = []
    counter = list(range(len(home_goalie_stats))) # counter for number of goalies that played

    for g in counter:
        # create dictonary for goalie
        home_goalie = {'date':game_date, 'game_id':game_id, 'team':home_goalie_team,
                       'goalie_name':home_goalie_names[g], 'goalie_id':home_goalie_id[g],
                       'is_home_team':True}
        home_goalie.update(home_goalie_stats[g])
        
        home_goalies.append(home_goalie)

    # make away goalie list. for loop needed as there could be more than 2 goalies playing in 1 game
    away_goalies = []
    counter = list(range(len(away_goalie_stats))) # counter for number of goalies that played

    for g in counter:
        # create dictonary for goalie
        away_goalie = {'date':game_date, 'game_id':game_id, 'team':away_goalie_team,
                       'goalie_name':away_goalie_names[g], 'goalie_id':away_goalie_id[g],
                       'is_home_team':False}
        away_goalie.update(away_goalie_stats[g])
        
        away_goalies.append(away_goalie)


    # Merge the two lists
    goalie_stats = away_goalies + home_goalies

    return goalie_stats
#%%
def scrape_game_info(game_id:int) -> Dict:
    """
        returns an dictionary with game information for the game_id provided
        Refer to: https://github.com/dword4/nhlapi on how to use the NHL API

        Arguments
            game_id (int): game id we are retrieving data for

        Returns
            Dict: Dictionary with information for the game_id provided
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

    # retrieve outcome
    if json_data['liveData']['linescore']['hasShootout']==False:
        if json_data["liveData"]["boxscore"]['teams']['home']['teamStats']['teamSkaterStats']['goals'] > json_data["liveData"]["boxscore"]['teams']['away']['teamStats']['teamSkaterStats']['goals']:
            home_team_win = True
        if json_data["liveData"]["boxscore"]['teams']['home']['teamStats']['teamSkaterStats']['goals'] < json_data["liveData"]["boxscore"]['teams']['away']['teamStats']['teamSkaterStats']['goals']:
            home_team_win = False
    if json_data['liveData']['linescore']['hasShootout']==True:
        if json_data['liveData']['linescore']['shootoutInfo']['home']['scores'] > json_data['liveData']['linescore']['shootoutInfo']['away']['scores']:
            home_team_win = True
        if json_data['liveData']['linescore']['shootoutInfo']['home']['scores'] < json_data['liveData']['linescore']['shootoutInfo']['away']['scores']:
            home_team_win = False

    # Starting goalies
    # spot checked a few APIs and it seems like the starting goalie will be listed last in the json
    # file if he was pulled. The goalie that finishes the game will be listed first (0).
    home_team_starting_goalie_id = json_data["liveData"]['boxscore']['teams']['home']['goalies'][-1]
    away_team_starting_goalie_id = json_data["liveData"]['boxscore']['teams']['away']['goalies'][-1]
    home_team_starting_goalie_name = \
    json_data["liveData"]['boxscore']['teams']['home']['players']['ID' + str(home_team_starting_goalie_id)][
        'person']['fullName']
    away_team_starting_goalie_name = \
    json_data["liveData"]['boxscore']['teams']['away']['players']['ID' + str(away_team_starting_goalie_id)][
        'person']['fullName']
    if game_id == 2020020215: # manually entering incorrect input data in NHL API for this game
        game_info = {'date':game_date, 'game_id':game_id, 'home_team':home_team, 'away_team':away_team,
                     'home_team_win':False, 'home_goalie_id':home_team_starting_goalie_id,
                     'away_goalie_id':away_team_starting_goalie_id,
                     'home_goalie_name':home_team_starting_goalie_name,
                     'away_goalie_name':away_team_starting_goalie_name}
    else:
        game_info = {'date':game_date, 'game_id':game_id, 'home_team':home_team, 'away_team':away_team,
                     'home_team_win':home_team_win, 'home_goalie_id':home_team_starting_goalie_id,
                     'away_goalie_id':away_team_starting_goalie_id,
                     'home_goalie_name':home_team_starting_goalie_name,
                     'away_goalie_name':away_team_starting_goalie_name}
    return game_info
#%%
def retrieve_team(game_id: int, home: bool) -> str:
    """
    retrieves the team abbreviation playing in an NHL game

    Arguments
        game_id (int): game id we are retrieving data for
        home (bool): if True retrieves the home team, False retrieves away
    
    Returns
        team (str): team abbreviation
    """

    url = f'https://statsapi.web.nhl.com/api/v1/game/{str(game_id)}/feed/live'
    resp = requests.get(url)
    json_data = json.loads(resp.text)

    if home:
        team = json_data['gameData']['teams']['home']['abbreviation']
    else:
        team = json_data['gameData']['teams']['away']['abbreviation']

    return team
#%%
def retrieve_date(game_id: int) -> dt.datetime:
    """
    retrieves the date an NHL game was played
    ...
    Parameters
    ----------
    game_id: int
        game id we are retrieving data for
    Returns
    -------
    date: dt.datetime
        date that NHL game was played
    """
    url = f'https://statsapi.web.nhl.com/api/v1/game/{str(game_id)}/feed/live'
    resp = requests.get(url)
    json_data = json.loads(resp.text)

    date = json_data['gameData']['datetime']['dateTime']
    date = dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')

    return date

#%%
def get_starting_goalies(home_abv: str, away_abv: str, date: str) -> str:
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

#%%
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
    url = 'https://statsapi.web.nhl.com/api/v1/teams'
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

#%%
ids_test = get_game_ids(20182019)
#%%
team_test = scrape_team_stats(ids_test[60])
#%%
players_test = scrape_player_stats(ids_test[60])
#%%
goalies_test = scrape_goalie_stats(ids_test[66])
#%%
game_test = scrape_game_info(ids_test[66])
