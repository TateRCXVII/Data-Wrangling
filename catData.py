import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.max_columns = 100


df = pd.read_csv(
    'https://www.dropbox.com/s/q9en0ls1xry42g2/Mental%20Illness%20Survey%201.csv?dl=1')
df = df.drop(index=0)
df.head()

# make income brackets (on 'Household Income') for easier analysis
df['Household Income'] = df['Household Income'].replace(
    ['Prefer not to answer', 'Rather not say'], 'Unknown')
df['Household Income'] = df['Household Income'].replace(
    ['$0 - $9,999', '$10,000 - $24,999'], 'Low')
df['Household Income'] = df['Household Income'].replace(
    ['$25,000 - $49,999', '$50,000 - $74,999'], 'Medium')
df['Household Income'] = df['Household Income'].replace(
    ['$75,000 - $99,999', '$100,000+'], 'High')

print(df['Household Income'].value_counts())
print(df['Household Income'])
