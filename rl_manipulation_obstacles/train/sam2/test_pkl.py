import pickle


with open("action_preprocessing_BERT_cat2.pkl","rb") as f:
    stats = pickle.load(f)


print(stats.keys())