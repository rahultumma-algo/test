import pandas as pd

# Load the CSV file
df = pd.read_csv('balakanda.csv')

# Iterate over each row and print the content and explanation
# for index, row in df.iterrows():
    # explanation = row['explanation']
explanation = df.loc[0, 'explanation']

print("Explanation:", explanation)
print()
