#%%
from bs4 import BeautifulSoup
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import sys
from typing import List, Tuple
import requests
import datetime as dt
import re
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
#%%
def get_nhl_dates(season: int) -> List[dt.datetime]:
    """
    returns a list of all dates NHL games were played in the specified season
    ...
    Parameters
    ----------
    season: int
        the two years of the season that nhl dates will be retrieved for (ex. 20192020)
    Returns
    -------
    list[dt.datetime]
        a list of datetime objects
    """
    # Determine the two years that NHL games were played for that season
    year1: int = int(str(season)[:4])
    year2: int = int(str(season)[4:])

    # Form the url and get response from hockey-reference.com
    url: str = f'https://www.hockey-reference.com/leagues/NHL_{year2}_games.html'
    resp: type = requests.get(url)

    # Find all the days games were played for year1 and year 2.
    days1: List[str] = re.findall(f'html">({year1}.*?)</a></th>', resp.text)
    days2: List[str] = re.findall(f'html">({year2}.*?)</a></th>', resp.text)
    days: List[str] = days1 + days2

    # Remove duplicates and convert strings to datetime
    days = remove_duplicates(days)
    dates: List[dt.datetime] = [dt.datetime.strptime(d, '%Y-%m-%d') for d in days]

    print(f'Number of days NHL regular season played in {season}: ', len(dates))
    return dates
#%%
def nhl_games_date(date: dt.datetime) -> List[dict]:
    """
    creates a list of NhlGame dictionaries for all games played on the date provided
    ...
    Parameters
    ----------
    date: dt.datetime
        datetime object for which we want to
    Returns
    -------
    list[NhlGame]
        a list of NhlGame objects
    """
    games = []

    # retrieve the covers.com webpage for the date provided
    date = date.strftime('%Y-%m-%d')
    url = f'https://www.covers.com/sports/nhl/matchups?selectedDate={date}'
    resp = requests.get(url)

    # parse the page, and retrieve all the game boxes on the page
    scraped_games = BeautifulSoup(resp.text, features='html.parser').findAll('div', {'class': 'cmg_matchup_game_box'})

    # iterate through all the game boxes and retrieve required information for NhlGame object
    for g in scraped_games:
        game_id = g['data-event-id'] # game_id
        h_abv = g['data-home-team-shortname-search'] # home_team
        a_abv = g['data-away-team-shortname-search'] # away_team
        h_ml = g['data-game-odd'] # home moneyline

        try:
            h_score = g.find('div', {'class': 'cmg_matchup_list_score_home'}).get_text(strip=True) # home score
            a_score = g.find('div', {'class': 'cmg_matchup_list_score_away'}).get_text(strip=True) # away score
        except:  # If a score cannot be found leave as blank
            h_score = ''
            a_score = ''

        game = {'date':date, 'game_id':game_id, 'home_team':h_abv,
                'away_team':a_abv, 'home_ml':h_ml, 'home_score':h_score,
                'away_score':a_score}

        games.append(game)

    return games
#%%
def sportsbook_accuracy(game_data: List[dict]) -> float:
    """
    calculates the accuracy score of the sportsbooks for the provided games
    ...
    Parameters
    ----------
    game_data: List[Dict]
        list of NhlGame objects for which we want to calculate the accuracy score for
    Returns
    -------
    accuracy: float
        the accuracy of the sportsbooks predictions
    outcomes: List[Boolean]
        True if the home team wins
    predictions: List[Boolean]
        True if the sportsbook had the home team favoured
    probabilities: List[float]
        the implied probability of the home team winning based on the moneyline odds
    """
    outcomes = []  # The actual outcome of the game. True if the home team wins
    predictions = []  # The sportsbook's "prediction". True if the home team was favoured.
    probabilities = []  # The implied probabilities determined from the moneyline odds

    for d in game_data:
        moneyline = int(d['home_ml'])
        home_score = int(d['home_score'])
        away_score = int(d['away_score'])

        if moneyline == 100:
            # We will exclude tossups for the calibration curve
            continue

        # Convert moneyline odds to their implied probabilities
        if moneyline < 0:
            probabilities.append(moneyline / (moneyline - 100))
        elif moneyline > 100:
            probabilities.append(100 / (moneyline + 100))

        outcomes.append(home_score > away_score)
        predictions.append(moneyline < 0)

    accuracy = 100 * accuracy_score(outcomes, predictions)

    return accuracy, outcomes, predictions, probabilities
#%%
def cal_curve(data, bins):
    """
    creates a calibration curve. X axis is the implied probability while the y axis is the percentage
    of time that prediction was correct
    ...
    Parameters
    ----------
    data: List[Tuple[List[Boolean], List[Boolean], List[Float], str]]
        the data that will be used to create the calibration curve. (Outcomes, Predictions, Probabilities, Name)
    Returns
    -------
    accuracy: float
        the accuracy of the sportsbooks predictions
    outcomes: List[Boolean]
        True if the home team wins
    predictions: List[Boolean]
        True if the sportsbook had the home team favoured
    probabilities: List[float]
        the implied probability of the home team winning based on the moneyline odds
    """

    fig = plt.figure(1, figsize=(12, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    for y_test, y_pred, y_proba, name in data:
        brier = brier_score_loss(y_test, y_proba)
        print("{}\tAccuracy:{:.4f}\t Brier Loss: {:.4f}".format(
            name, accuracy_score(y_test, y_pred), brier))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, label="%s (%1.4f)" % (name, brier))
        ax2.hist(y_proba, range=(0, 1), bins=bins, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    fig.savefig('sb_evaluation.png')
#%% get dates that nhl games were played on in the 2018/2019 season
dates = get_nhl_dates(20182019)
#%%
games = [] # will hold NhlGame objects for all games played in the 2018/2019 season
#%%
for d in dates:
    games += nhl_games_date(d)

accuracy, outcomes, predictions, probabilities = sportsbook_accuracy(games)
data = [(outcomes, predictions, probabilities, 'Sportsbook')]
#%% pickle results
with open('/Users/patrickpetanca/projects/nhl_model/data/baseline.pkl', 'wb') as f:
    pickle.dump((outcomes, predictions, probabilities), f)
#%% create calibration curve and save figure in data folder
cal_curve(data,15)