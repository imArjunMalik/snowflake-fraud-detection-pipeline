import pandas as pd

# Load the generated dataset
df = pd.read_csv('synthetic_account_opening_data.csv')

# Check for unique genuine entries
genuine_entries = df.drop_duplicates(subset=['SSN', 'First_Name', 'Last_Name', 'Phone_Number', 'Email_Address'])

# Print the number of unique genuine entries
print(f"Number of unique genuine entries: {len(genuine_entries)}")

# Optionally, display a few rows to manually verify
print(genuine_entries.head())
