import pandas as pd
import glob
import os

# Mapping of league codes to descriptive names
league_map = {
    'SP1': 'LaLiga',
    'E0': 'PremierLeague',
    'D1': 'Bundesliga',
    'I1': 'SerieA',
    'F1': 'Ligue1'
}

# Features to keep (Normalization)
# We keep identifiers + the metrics required for the model
keep_columns = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTR', # Identifiers & Target
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR'     # Match Stats
]

output_dir = 'clean_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting Data Normalization...")

for code, name in league_map.items():
    # Find all files for this league (e.g., raw_data/SP1_*.csv)
    files = glob.glob(f"raw_data/{code}_*.csv")
    
    if not files:
        print(f"No files found for {name} ({code})")
        continue
        
    print(f"Processing {name} ({code})... Found {len(files)} files.")
    
    dfs = []
    for f in files:
        try:
            # Read CSV
            df = pd.read_csv(f, on_bad_lines='skip')
            
            # Ensure Date column is standard
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Select only relevant columns if they exist
            # We filter columns that actually exist in the dataframe to avoid KeyErrors
            existing_cols = [c for c in keep_columns if c in df.columns]
            df_subset = df[existing_cols]
            
            dfs.append(df_subset)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    
    if dfs:
        # Combine all seasons
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by Date
        if 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values('Date')
        
        # Save to single file
        output_path = f"{output_dir}/{name}_Combined.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"  Saved combined data to {output_path} ({len(combined_df)} matches)")

print("\nNormalization Complete!")
