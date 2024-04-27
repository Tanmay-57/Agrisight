import pandas as pd

try:
  # Try semicolon delimiter first
  data = pd.read_csv("testing.csv", delimiter=";")
except pd.errors.ParserError:
  try:
    # If semicolon fails, try comma delimiter
    data = pd.read_csv("testing.csv", delimiter=",")
  except pd.errors.ParserError:
    print("Error: Could not parse CSV using semicolon or comma delimiters.")

# Process data if successfully read
if 'data' in locals():  # Check if data variable exists after reading
  # Rest of your code to process data
  print(data.head())  # Print the first few rows for verification
