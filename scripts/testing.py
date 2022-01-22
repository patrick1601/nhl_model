import pickle

with open('/Users/patrickpetanca/projects/nhl_model/data/predictions.pkl', 'rb') as f:
    p = pickle.load(f)
print(p)