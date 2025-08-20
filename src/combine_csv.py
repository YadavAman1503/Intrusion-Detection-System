import pandas as pd
import glob
import os

def combine_unsw_files():
    # Step 1: Point to the right folder from src/
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    all_files = glob.glob(os.path.join(data_dir, "Data.csv"))

    # Step 2: Combine all matching CSV files
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # Step 3: Save combined file
    output_path = os.path.join(data_dir, "Data.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Combined CSV saved as {output_path}")

if __name__ == "__main__":
    combine_unsw_files()
