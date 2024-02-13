import os
from tqdm import tqdm
import pandas as pd

input_folder = '../../files/2020-06-19-dgarchive_full'
output_folder = '../../files/dga_30K'\

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to randomly select up to 10,000 rows from a CSV file
def random_sample_csv(input_file, output_file, num_rows=30000):
    df = pd.read_csv(input_file)
    if len(df) <= num_rows:
        # If the file has 10,000 or fewer rows, keep all rows
        df.to_csv(output_file, index=False)
    else:
        # Randomly select up to 10,000 rows
        random_sample = df.sample(n=num_rows, random_state=42)
        random_sample.to_csv(output_file, index=False)


# Loop through all the files in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.csv'):
        in_file = os.path.join(input_folder, filename)
        out_file = os.path.join(output_folder, filename)

        # Check if the output file already exists and delete it if it does
        if os.path.exists(out_file):
            os.remove(out_file)

        # Call the random_sample_csv function to select up to 10,000 rows
        random_sample_csv(in_file, out_file, num_rows=30000)

folder_path = '../../files/dga_30K'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# first_column = pd.Series()
#
# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)
#     first_column = pd.concat([first_column, df.iloc[:, 0]])
#
# merged_file_path = '../../files/dga_20K/dga_10K_concat.csv'
# first_column.to_csv(merged_file_path, index=False)