import sqlite3
import pandas as pd

conn = sqlite3.connect("automotive_data.db")

query = """
SELECT vehicle_id, speed_kmh, brake_pressure
FROM telemetry
WHERE brake_pressure > 80
LIMIT 10
"""

result = pd.read_sql(query, conn)

print(result)