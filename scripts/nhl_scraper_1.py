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
#%% get game ids from nhl.com API
def get_game_ids (season: int) -> List[int]:
    '''
    Retrieves all of the game ids for the provided season

    Arguments:
        season (int): the season for which you want to retrieve game ids (ex: 20192020)

    Returns:
        List[int]: a list containing all regular season game ids for that season
    '''
