'''
scrape data and create dictionaries from nhl.com
'''
import pickle
from typing import List
import pandas as pd
import nhl_scraper_1

#%% show full columns on dataframes
pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", 25)

#%% scrape game ids
def pull_game_ids(first_year: int=2010, last_year: int=2020) -> List[int]:
    """
    pulls all nhl game ids between the specified dates
    ...
    Parameters
    ----------
    first_year: int
        first year to retrieve game ids
    last_year: int
        last year to retrieve game ids
    Returns
    -------
    game_ids: str
    """
    #create a list of years for which we want data
    years = list(range(first_year, last_year))

    #create year for the get_game_ids() function argument in the format 20192020
    game_ids_url_years = []

    for i in years:
        j = str(i) + str(i+1)
        game_ids_url_years.append(j)

    #run for loop to retrieve game IDs for all seasons required
    ids = []
    for i in game_ids_url_years:

        if len(ids) % 500 == 0:  # Progress bar
            print(str(len(ids) / len(game_ids_url_years) * 100) +
                  ' percent done retrieving game ids.')

        try:
            ids = ids + nhl_scraper_1.get_game_ids(i)

        except KeyError:
            print(str('*************Not able to retrieve: ' +
                      str(i) +
                      ' games due to KeyError************'))
            continue

    return ids

#%% scrape team stats
def pull_team_stats(ids: List[int]) -> List[dict]:
    """
    pulls all team stats for the provided game ids
    ...
    Parameters
    ----------
    ids: List[int]
        list of game ids to pull team stats for
    Returns
    -------
    team_stats: List[dict]
        list of NhlTeam objects
    """

    # retrieve game by game stats for every game in the ids list
    stats = []

    for i in ids:
        stats_i = nhl_scraper_1.scrape_team_stats(i)
        stats += stats_i

        if len(stats) % 500 == 0:  # Progress bar
            print(str(0.5 * len(stats) /
                      len(ids) * 100) +
                  ' percent done retrieving game data/stats.')

    return stats

#%% scrape goalie stats
def pull_goalie_stats(ids: List[int]) -> List[dict]:
    """
        pulls all goalie stats for the provided game ids
        ...
        Parameters
        ----------
        game_ids: List[int]
            list of game ids to pull team stats for
        Returns
        -------
        goalie_stats: List[dict]
            list of nhl goalie dictionaries each entry
            represents 1 game played by 1 goalie
        """

    goalie_stats_list=[]
    for i in ids:
        goalies_i = nhl_scraper_1.scrape_goalie_stats(i)
        goalie_stats_list += goalies_i

        # progress bar todo fix progress bar to account for more goalies than game ids
        if len(goalie_stats_list) % 250 == 0:
            print(str(0.5 * len(goalie_stats_list) /
                      len(ids) * 100) +
                  ' percent done retrieving goalie data.')

    return goalie_stats_list

#%% scrape game info
def pull_game_info(ids: List[int]) -> List[dict]:
    """
    pulls all game_info for the provided game ids
    ...
    Parameters
    ----------
    ids: List[int]
        list of game ids to pull team stats for
    Returns
    -------
    games_info: List[dict]
        list of dictionaries
    """

    # retrieve game by game info for every game in the game_ids list
    games_info = []

    for i in ids:
        game_i = nhl_scraper_1.scrape_game_info(i)
        games_info.append(game_i)

        if len(games_info) % 500 == 0:  # Progress bar
            print(str(len(games_info) /
                      len(ids) * 100) +
                  ' percent done retrieving game data/stats.')
    return games_info

#%% scrape game ids
game_ids = pull_game_ids(first_year=2010, last_year=2020)
with open('/Users/patrickpetanca/projects/nhl_model/data/game_ids.pkl', 'wb') as f:
    pickle.dump(game_ids, f)
#%% scrape team stats
team_stats = pull_team_stats(game_ids)
with open('/Users/patrickpetanca/projects/nhl_model/data/team_stats.pkl', 'wb') as f:
    pickle.dump(team_stats, f)
#%% scrape goalie stats
goalie_stats = pull_goalie_stats(game_ids)
with open('/Users/patrickpetanca/projects/nhl_model/data/goalie_stats.pkl', 'wb') as f:
    pickle.dump(goalie_stats, f)
#%% scrape game info
game_info = pull_game_info(game_ids)
with open('/Users/patrickpetanca/projects/nhl_model/data/games_info.pkl', 'wb') as f:
    pickle.dump(game_info, f)
    