## Kaggle Python Tutorial on Machine Learning


## Chapter 1: Getting Started with Python


## Exercise 1: How it works

# Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

# Compute y = 6 * 9 and print the result
y = 6 * 9
print(y)


## Exercise 2: Get the Data with Pandas

# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())


## Exercise 3: Understanding your data

# Use .describe() to see summaries of rows and columns in the train and test datasets
train.describe()
test.describe()

# Print the dimensions of the datasets
print(train.shape)
print(test.shape)

# Answer: The training set has 891 observations and 12 variables, count for Age is 714.


## Exercise 4: Rose vs Jack, or Female vs Male

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))


## Exercise 5: Does age play a role?

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] <18] = 1
train["Child"][train["Age"] >= 18] = 0

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


## Exercise 6: First Prediction

# Create a copy of test: test_one
test_one = test.copy()

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"] == "female"] = 1
print(test_one["Survived"])