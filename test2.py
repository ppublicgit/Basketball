import pandas as pd
import numpy as np


df = pd.read_csv("test.csv")
df.set_index(["game_id", "home_flag"], inplace=True)
df.sort_index(inplace=True)


columns = ["wl%", "asts", "rebs", "orebs",
           "tovs", "fga", "fg%", "3pa", "3p%",
           "fta", "ft%", "pfs", "net_score", "won"]
temp_dict = {}
for col in columns:
    temp_dict[col] = []
for i in range(0, len(df), 2):
    away = df.iloc[i]
    home = df.iloc[i+1]
    for col in columns:
        new_val = home[col] - away[col]
        if col == "net_score":
            new_val /= 2
        temp_dict[col].append(new_val)

new_df = pd.DataFrame(data=temp_dict, columns=columns)
