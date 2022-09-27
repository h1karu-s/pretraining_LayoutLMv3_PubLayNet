
import pickle
import time

start = time.time()
with open("../data/preprocessing_shared/encoded_dataset_mim_30000.pkl", 'rb') as f:
    data = pickle.load(f)
t = time.time() - start
print(len(data), flush=True)
print(t)


