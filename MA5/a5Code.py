from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in the data
heart_data = pd.read_csv("./data/heart.csv")
o2_data = pd.read_csv("./data/o2Saturation.csv")

# in this, we are comparing age and heart health to other factors, like oxygen saturation and resting blood pressure, fasting blood sugar, and cholesterol

# let's add the o2 data to the heart data
# add the header "o2Saturation" to the o2_data since there is no header
o2_data.columns = ["o2Saturation"]
heart_data["o2Saturation"] = o2_data["o2Saturation"]

# let's look at the data
print("Head:", heart_data.head())
print("Tail:", heart_data.tail())
print("Describe:", heart_data.describe())

# validate the data
print("Shape:", heart_data.shape)

# the headers are a little confusing so lets rename them according to this dataset description
# Age : Age of the patient

# Sex : Sex of the patient

# exang: exercise induced angina (1 = yes; 0 = no)

# ca: number of major vessels (0-3)

# cp : Chest Pain type chest pain type

# Value 1: typical angina
# Value 2: atypical angina
# Value 3: non-anginal pain
# Value 4: asymptomatic
# trtbps : resting blood pressure (in mm Hg)

# chol : cholestoral in mg/dl fetched via BMI sensor

# fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

# rest_ecg : resting electrocardiographic results

# Value 0: normal
# Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# thalach : maximum heart rate achieved

# target : 0= less chance of heart attack 1= more chance of heart attack


# rename the headers
heart_data.rename(
    columns={
        "Age": "age",
        "exang": "Exercise Induced Angina",
        "ca": "Number of Major Vessels",
        "cp": "Chest Pain Type",
        "trtbps": "Resting Blood Pressure",
        "chol": "Cholesterol",
        "fbs": "Fasting Blood Sugar",
        "restecg": "Resting ECG",
        "thalachh": "Max Heart Rate",
        "exng": "Exercise Induced Angina",
        "oldpeak": "ST Depression",
        "slp": "Slope",
        "caa": "Number of Major Vessels",
        "thall": "Thalium Stress Test",
        "target": "Heart Attack",
    },
    inplace=True,
)

# let's look at the data
print("Head:", heart_data.head())

bins = range(100, 600, 50)

# Create labels for cholesterol bins
labels = ['{}-{}'.format(b, b+49) for b in bins[:-1]]
heart_cholestrol_cont_table = pd.cut(heart_data['Cholesterol'], bins=bins, labels=labels)

# create bins for age for every 5 years starting at 29 and ending at 77
bins = range(29, 78, 5)

# Create labels for age bins
labels = ['{}-{}'.format(b, b+4) for b in bins[:-1]]
heart_age_cont_table = pd.cut(heart_data['age'], bins=bins, labels=labels)

# Let's analyze and compare the data
# we want to look at the correlation between age and heart attack probability first
# we can use the pearson correlation coefficient to see how correlated the data is
# the pearson correlation coefficient is a measure of the linear correlation between two variables X and Y
# it has a value between -1 and 1, where 1 is total positive linear correlation, 0 is no linear correlation, and -1 is total negative linear correlation
# the formula for the pearson correlation coefficient is:
# r = (sum((x - mean(x)) * (y - mean(y)))) / (sqrt(sum((x - mean(x))^2)) * sqrt(sum((y - mean(y))^2)))
# where x is the first set of data and y is the second set of data
# we can use the scipy.stats.pearsonr function to calculate the pearson correlation coefficient and the p-value for testing non-correlation

# compare the age and heart attack probability in a line graph
# heart attack only has values of 0 and 1. Let's use a logistic regression to get the probability of a heart attack
# we can use the scipy.stats.logistic function to get the probability of a heart attack
# the logistic function is a sigmoid function that is used to model growth
# it has a characteristic "S" shape
# the logistic function is defined by the formula:
# f(x) = 1 / (1 + e^(-x))
# where x is the input value
# we can use the scipy.stats.logistic function to get the probability of a heart attack

# get the age and heart attack probability using logistic regression
cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['age'])

stat, p, dof, expected = stats.chi2_contingency(cont_table)

# The null hypothesis for the chi-square test is that there is no association 
# between heart attack probability and age. If the p-value is less than your 
# chosen significance level (e.g. 0.05), you can reject the null hypothesis and 
# conclude that there is a statistically significant association between heart 
# attack probability and age.

print('stat=%.3f, p=%.3f' % (stat, p))
print('dof=%d' % dof)
print('expected=%s' % expected)

# do the same but for the resting blood pressure
cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['Resting Blood Pressure'])

stat, p, dof, expected = stats.chi2_contingency(cont_table)

print('stat=%.3f, p=%.3f' % (stat, p))
print('dof=%d' % dof)
print('expected=%s' % expected)

# do the same but for cholesterol
cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['Cholesterol'])

stat, p, dof, expected = stats.chi2_contingency(cont_table)

print('stat=%.3f, p=%.3f' % (stat, p))
print('dof=%d' % dof)
print('expected=%s' % expected)

cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['o2Saturation'])

stat, p, dof, expected = stats.chi2_contingency(cont_table)

print('stat=%.3f, p=%.5f' % (stat, p))
print('dof=%d' % dof)
print('expected=%s' % expected)


# calculate the effect size for the age and heart attack probability
# and compare it to the effect size for the o2Saturation and heart attack probability

# let's plot the data and compare

heart_o2_cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['o2Saturation'])

heart_age_cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['age'])


# for cholesterol


heart_cholestrol_cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['Cholesterol'])


# break up cholesteral into groups of 50
# heart_cholestrol_cont_table = pd.crosstab(heart_data['Heart Attack'], heart_data['Cholesterol'] // 50)


# use a stacked bar chart to compare the data
# we can use the pandas.DataFrame.plot.bar function to plot the data

# plot the data
heart_o2_cont_table.plot.bar(stacked=True)
plt.show()

heart_age_cont_table.plot.bar(stacked=True)
plt.show()

heart_cholestrol_cont_table.plot.bar(stacked=True)
plt.show()

