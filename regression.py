import os, json, torch
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from dataset import BlendShapeDataset, Set

Set.RETURN_FULL = True
MODEL = 'meta-llama/Llama-2-7b-chat-hf' # Any kind of model is fine here because we wont use it

INTONATIONS = ["rising", "falling"]
FILLER_WORDS = ["none", "uh", "um"]
PRE_HEDGE = ["none", "ithink", "maybe"]
POST_HEDGE = ["none", "butimnotsure", "idontknow"]
HEADERS = ["intonation", "fillerWord", "preHedge", "postHedge", "prePause", "postPause", "confidence"]
DIRECTORY = "data/SONA"

INPUT_COLUMNS = ["confidence", "intonation_falling", "intonation_rising", "fillerWord_none", "fillerWord_uh", "fillerWord_um", "preHedge_ithink", "preHedge_maybe", "preHedge_none", "postHedge_butimnotsure", "postHedge_idontknow", "postHedge_none"]

dataset = BlendShapeDataset("data", "SONA", 600, 1, False, MODEL)
dataset.setup()
train_data = dataset.train_dataloader()


dataset = []
for raw in train_data:
    d = []
    for val in [raw[header][0] for header in HEADERS]:
        try:
            d.append(val.item())
        except:
            d.append(val)
    dataset.append(d)

dataset = pd.DataFrame(data=dataset, columns=HEADERS)
files = [file for file in os.listdir(DIRECTORY) if file.endswith(".json")]


dataset = []
for raw in [json.load(open(os.path.join(DIRECTORY, file), "r")) for file in files]:
    dataset.append(
        [raw[header] for header in HEADERS]
    )

dataset = pd.DataFrame(data=dataset, columns=HEADERS)
df_encoded = pd.get_dummies(dataset, columns=['intonation', 'fillerWord', 'preHedge', 'postHedge'])

input_data = df_encoded[INPUT_COLUMNS].to_numpy(np.float32)
output_data = np.log(df_encoded[["prePause"]].to_numpy(np.float32))
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=42)

pre_regressor = MLPRegressor((16, 16, 16), activation="tanh").fit(X_train, y_train)
dump(pre_regressor, "pre_regressor.jlib")

input_data = df_encoded[INPUT_COLUMNS].to_numpy(np.float32)
output_data = np.log(df_encoded[["postPause"]].to_numpy(np.float32))
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=42)

post_regressor = MLPRegressor((16, 16, 16), activation="tanh").fit(X_train, y_train)
dump(post_regressor, "post_regressor.jlib")