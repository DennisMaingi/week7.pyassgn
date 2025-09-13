# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# improve plot aesthetics with seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset

try:
    # Load dataset 
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Dataset info (data types, non-null counts)
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Clean dataset: drop rows with missing values (or fillna if you prefer)
    df_clean = df.dropna()
    print(f"\nRows before cleaning: {len(df)}, after cleaning: {len(df_clean)}")

except FileNotFoundError:
    print("Error: Dataset file not found.")
except pd.errors.EmptyDataError:
    print("Error: Dataset file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic statistics of numerical columns:")
print(df_clean.describe())

# Grouping example: mean sepal_length per species
grouped = df_clean.groupby('species')['sepal_length'].mean()
print("\nMean sepal_length per species:")
print(grouped)

# Observations
print("\nObservations:")
print("- Versicolor has the average sepal length around", round(grouped['versicolor'], 2))
print("- Setosa has the shortest sepal length on average")
print("- Virginica has the longest sepal length on average")

# Task 3: Data Visualization

# 1. Line chart (trends over time)
# Iris dataset doesn't have time, so let's plot mean sepal_length by species in a line chart
plt.figure(figsize=(8, 5))
grouped.plot(kind='line', marker='o')
plt.title("Mean Sepal Length by Species (Line Chart)")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length")
plt.grid(True)
plt.show()

# 2. Bar chart (comparison across categories)
plt.figure(figsize=(8, 5))
grouped.plot(kind='bar', color=['#FF9999','#66B2FF','#99FF99'])
plt.title("Mean Sepal Length by Species (Bar Chart)")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length")
plt.xticks(rotation=0)
plt.show()

# 3. Histogram (distribution of a numerical column)
plt.figure(figsize=(8, 5))
plt.hist(df_clean['sepal_length'], bins=15, color='purple', alpha=0.7)
plt.title("Distribution of Sepal Length (Histogram)")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (relationship between two numerical columns)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='sepal_length', y='petal_length', hue='species')
plt.title("Sepal Length vs Petal Length (Scatter Plot)")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title='Species')
plt.show()
