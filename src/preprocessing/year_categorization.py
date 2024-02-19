input_folder = '../../files/2020-06-19-dgarchive_full'
output_folder = '../../files/dga_full_by_year'

import os
import pandas as pd

def truncate_csv_by_year(input_folder, output_folder):
    # Iterate through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            # Construct the full path of the CSV file
            input_file = os.path.join(input_folder, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(input_file, header=None)

            # # Convert the 'date' column to datetime type
            print(df[2].dtype)
            df[2] = pd.to_datetime(df[2], errors='coerce', utc=True)
            print(df[2].dtype)

            # Group by date and create smaller DataFrames
            grouped_data = df.groupby(df[2].dt.year)

            # Save smaller CSV files for each date
            for year, group in grouped_data:
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{year}.csv")
                group.to_csv(output_file, index=False, header=None)

if __name__ == "__main__":

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Call the function to truncate CSV files in the input folder
    truncate_csv_by_year(input_folder, output_folder)
