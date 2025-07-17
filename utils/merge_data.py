import pandas as pd
import xarray as xr

def load_gfw_data(csv_path):
    """
    Load Global Fishing Watch CSV data and add binary labels.
    Label = 1 if fishing hours > 0, else 0.
    """
    gfw = pd.read_csv(csv_path)
    gfw['date'] = pd.to_datetime(gfw['date'])
    gfw['label'] = (gfw['fishing_hours'] > 0).astype(int)
    gfw = gfw[['lat', 'lon', 'date', 'label']]
    return gfw

def load_sst_data(nc_path):
    """
    Load NOAA SST data from NetCDF file.
    Rounds coordinates to 0.1° to match GFW grid.
    """
    sst_ds = xr.open_dataset(nc_path)
    
    # You may need to inspect this variable name and adjust if needed
    sst = sst_ds['analysed_sst']
    
    # Convert SST data to DataFrame
    sst_df = sst.to_dataframe().reset_index()
    sst_df.rename(columns={"analysed_sst": "sst"}, inplace=True)
    
    # Round lat/lon to 0.1 degrees (to match GFW)
    sst_df['lat'] = sst_df['lat'].round(1)
    sst_df['lon'] = sst_df['lon'].round(1)
    
    # Extract date from time dimension
    sst_df['date'] = pd.to_datetime(sst_df['time'])
    
    return sst_df[['lat', 'lon', 'date', 'sst']]

def merge_datasets(gfw_path, sst_path, output_csv):
    """
    Merges GFW effort data and NOAA SST data into one CSV.
    """
    print("Loading fishing effort data...")
    gfw_df = load_gfw_data(gfw_path)
    
    print("Loading SST data...")
    sst_df = load_sst_data(sst_path)
    
    print("Merging datasets...")
    merged_df = pd.merge(gfw_df, sst_df, on=['lat', 'lon', 'date'], how='inner')
    
    print(f"Saving merged dataset to: {output_csv}")
    merged_df.to_csv(output_csv, index=False)

    print("✅ Merge complete. Sample:")
    print(merged_df.head())

if __name__ == "__main__":
    merge_datasets(
        gfw_path="../data/sample_gfw.csv",     # Replace with actual file path
        sst_path="../data/sample_sst.nc",      # Replace with actual SST NetCDF path
        output_csv="../data/merged_dataset.csv"
    )
