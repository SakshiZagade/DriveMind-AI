import sqlite3
from click import prompt
import pandas as pd
import requests
import json

# connect database
conn = sqlite3.connect("automotive_data.db")

def generate_sql(prompt):

    schema = """
Table: telemetry

Columns:
vehicle_id
session_id
timestamp
speed_kmh
acceleration
brake_pressure
steering_angle
engine_temp
rpm
fuel_level
latitude
longitude
road_type
weather
anomaly_label
"""

    system_prompt = f"""
You are an expert automotive data analyst.

Your task is to convert user questions into SQLite SQL queries.

Database schema:
{schema}

Rules:
- Use ONLY the columns listed in the schema.
- Do NOT create new column names.
- Return ONLY the SQL query.
"""

    full_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nSQL Query:"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": full_prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        sql = response.json()["response"]
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    return None

def fix_common_errors(sql):

    fixes = {
        "engine_temperature": "engine_temp",
        "speed": "speed_kmh",
        "fuel": "fuel_level",
        "brake": "brake_pressure"
    }

    for wrong, correct in fixes.items():
        sql = sql.replace(wrong, correct)

    return sql

sql = generate_sql(prompt)
sql = fix_common_errors(sql)