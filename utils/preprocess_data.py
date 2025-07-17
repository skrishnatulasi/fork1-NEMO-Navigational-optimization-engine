import pandas as pd
import os

# File paths
input_path = os.path.join('data', 'fishing_data.csv')
output_path = os.path.join('data', 'processed_fishing_data.csv')
cleaned_path = os.path.join('data', 'fishing_data_cleaned.csv')
excel_path = os.path.join('data', 'fishing_data.xlsx')

def try_read_csv(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

print('Loading data...')
df = try_read_csv(input_path, encoding='latin1', on_bad_lines='skip')

if df is None:
    print('Attempting to clean the CSV by removing problematic lines...')
    # Try to read line by line and save only good lines
    with open(input_path, 'r', encoding='latin1', errors='ignore') as infile, open(cleaned_path, 'w', encoding='latin1') as outfile:
        for line in infile:
            try:
                # Try parsing the line as a CSV row
                pd.read_csv(pd.compat.StringIO(line))
                outfile.write(line)
            except Exception:
                continue  # Skip bad lines
    df = try_read_csv(cleaned_path, encoding='latin1')

if df is None:
    print('Attempting to convert CSV to Excel and read from there...')
    try:
        # Try to convert to Excel
        temp_df = pd.read_csv(input_path, encoding='latin1', on_bad_lines='skip')
        temp_df.to_excel(excel_path, index=False)
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Failed to convert to Excel: {e}")
        raise RuntimeError('All attempts to read the data failed.')

print('First 5 rows:')
print(df.head())
print('\nColumns:')
print(df.columns)

# Basic EDA
print('\nMissing values per column:')
print(df.isnull().sum())

# Drop rows with missing essential values (timestamp, lat, lon, catch)
df = df.dropna(subset=['timestamp', 'latitude', 'longitude', 'catch'])

# Convert timestamp to datetime
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

# Reset index
df = df.reset_index(drop=True)

# Save processed data
df.to_csv(output_path, index=False)
print(f'Processed data saved to {output_path}')