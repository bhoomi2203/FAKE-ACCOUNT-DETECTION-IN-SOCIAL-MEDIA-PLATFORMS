# check_nan.py
import pandas as pd
import json
import numpy as np

# 1) CSV check
csv_path = 'train_csv.csv'
df = pd.read_csv(csv_path)
print("CSV shape:", df.shape)
print("CSV missing by column:\n", df.isnull().sum())

# 2) NLP JSON check (basic)
json_path = 'train_nlp.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("Loaded JSON samples:", len(data))

# If you construct feature DataFrame for NLP within your script, check the fields you extract:
# Example: check if expected fields exist / missing
fields_to_check = ['username','fullname','bio','captions','comments']  # adjust to your schema
counts = {field: 0 for field in fields_to_check}
for i, item in enumerate(data):
    for field in fields_to_check:
        if not item.get(field):  # empty string or None or missing
            counts[field] += 1
print("Missing counts in JSON fields:", counts)
