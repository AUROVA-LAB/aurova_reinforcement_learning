import pickle


with open("action_preprocessing.pkl","rb") as f:
    stats = pickle.load(f)


print(stats.keys())