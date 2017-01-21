import pandas as pd

# Code to write csv file into another csv file. (Only writes the first two rows)
#for chunk in pd.read_csv('GSE84727_normalisedBetas.csv', chunksize=100):
#    df = pd.DataFrame(chunk)
#    df.to_csv('first100Rows.csv')
#    break

# Count number of case/control
'''
df = pd.read_csv('disease_status.txt', sep='\t', header=None)
df = pd.DataFrame.transpose(df)

status_one_count = 0
status_two_count = 0

for index, row in df.iterrows():
    if row[0] == "disease_status: 1":
        status_one_count += 1
    else:
        status_two_count += 1

print status_one_count, status_two_count
print status_one_count + status_two_count
'''

count = 0
for chunk in pd.read_csv('GSE84727_normalisedBetas.csv', chunksize=100):
    df = pd.DataFrame(chunk)
    count += df['3998567027_R01C01'].count()

print count
