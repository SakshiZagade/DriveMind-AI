import pandas as pd
import sqlite3

# load dataset
data = pd.read_csv("data/automotive_telemetry_150k.csv")

# connect to database
conn = sqlite3.connect("automotive_data.db")

# store dataset as SQL table
data.to_sql("telemetry", conn, if_exists="replace", index=False)

print("Database created successfully!")
print("Table name: telemetry")