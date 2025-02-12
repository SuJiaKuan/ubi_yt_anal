import os
import json


# load a json file into a dict
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Save a dict into a json file
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
