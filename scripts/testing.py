import pickle
import pandas as pd
pd.set_option('display.max_columns', None)
with open('/Users/patrickpetanca/projects/nhl_model/data/predictions.pkl', 'rb') as f:
    predictions_df = pickle.load(f)

predictions_df.to_csv('/Users/patrickpetanca/projects/nhl_model/data/predictions.csv')