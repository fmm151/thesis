import os
import pandas as pd

def merge_csv_by_year(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dictionary to store DataFrames by year
    year_dataframes = {}

    # Iterate through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            # Construct the full path of the CSV file
            input_file = os.path.join(input_folder, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(input_file, header=None)

            # Extract the year from the filename (assuming the format is 'data_YEAR.csv')
            year = filename.split('_')[-1].split('.')[0]

            # Check if the year exists in the dictionary
            if year in year_dataframes:
                # Merge the current DataFrame with the existing one for the same year
                year_dataframes[year] = pd.concat([year_dataframes[year], df], ignore_index=True)
            else:
                # If the year is not in the dictionary, add the DataFrame for that year
                year_dataframes[year] = df

    # Save merged DataFrames to new CSV files
    for year, df in year_dataframes.items():
        output_file = os.path.join(output_folder, f"dga_{year}.csv")
        df.to_csv(output_file, index=False, header=None)

if __name__ == "__main__":
    # Specify the input folder containing CSV files and the output folder
    input_csv_folder = '../../files/dga_full_by_year'
    output_folder = '../../files/merged_dga_full_by_year'

    # Call the function to merge CSV files in the input folder by year
    merge_csv_by_year(input_csv_folder, output_folder)
