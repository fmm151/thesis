import pandas as pd
import matplotlib.pyplot as plt

# Your data (replace with your actual data)
data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'Accuracy': [0.934, 0.931, 0.927, 0.927, 0.915, 0.884, 0.889, 0.928, 0.928, 0.865],
    'Precision': [0.948, 0.955, 0.954, 0.956, 0.960, 0.958, 0.959, 0.962, 0.961, 0.956],  # Adjusted for missing value
    'Recall': [0.924, 0.92, 0.915, 0.914, 0.896, 0.845, 0.87, 0.917, 0.919, 0.816],
    'F1': [0.936, 0.937, 0.934, 0.935, 0.927, 0.898, 0.912, 0.939, 0.939, 0.88]  # Adjusted for missing values
}
#
# data = {
#     'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
#     'Accuracy': [0.946, 0.857, 0.628, 0.358, 0.276, 0.223, 0.203, 0.192, 0.175, 0.176],
#     'Precision': [0.985, 0.886, 0.799, 0.792, 0.794, 0.817, 0.839, 0.844, 0.851, 0.848],  # Adjusted for missing value
#     'Recall': [0.958, 0.834, 0.603, 0.349, 0.277, 0.229, 0.201, 0.183, 0.169, 0.174],
#     'F1': [0.953, 0.823, 0.553, 0.247, 0.164, 0.123, 0.107, 0.089, 0.077, 0.081]  # Adjusted for missing values
# }


# Create a DataFrame from the data
df = pd.DataFrame(data)

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Accuracy'], label='Accuracy', marker='o')
plt.plot(df['Year'], df['Precision'], label='Precision', marker='o')
plt.plot(df['Year'], df['Recall'], label='Recall', marker='o')
plt.plot(df['Year'], df['F1'], label='F1 Score', marker='o')

# Customize the plot
# plt.title('Metrics Over Years')
plt.xlabel('Year')
plt.ylabel('Values')
plt.xticks(df['Year'])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


import matplotlib.pyplot as plt

# Data
letters = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'R', 'H', 'D', 'L', 'U', 'C', 'M', 'F', 'Y', 'W', 'G', 'P', 'B', 'V',
           'K', 'X', 'Q', 'J', 'Z']
frequencies = [12.02, 9.10, 8.12, 7.68, 7.31, 6.95, 6.28, 6.02, 5.92, 4.32, 3.98, 2.88, 2.71, 2.61, 2.30, 2.11,
               2.09, 2.03, 1.82, 1.49, 1.11, 0.69, 0.17, 0.11, 0.10, 0.07]

# Sorting the data in descending order of frequencies
sorted_data = sorted(zip(letters, frequencies), key=lambda x: x[1], reverse=True)
sorted_letters, sorted_frequencies = zip(*sorted_data)

# Plotting
plt.figure(figsize=(14, 8))
plt.bar(sorted_letters, sorted_frequencies, color=(79 / 255, 121 / 255, 66 / 255))  # Custom RGB color
plt.xlabel('Γράμμα', fontsize=18)
plt.ylabel('Συχνότητα (%)', fontsize=18)
plt.xticks(fontsize=15)
plt.grid(True)
plt.show()

