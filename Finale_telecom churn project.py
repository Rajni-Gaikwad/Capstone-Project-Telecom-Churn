#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading "numpy" and "pandas" for manipulating numbers, vectors and data frames
# Loading "matplotlib.pyplot" and "seaborn" for data visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv(r'C:\Users\HP\Downloads\telecom_churn_data.csv')


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


# Getting the row names of the data frame using ".index"
df.index


# In[8]:


# Looking at the basic information about the data frame using ".info()"
df.info()


# In[9]:


import pandas as pd
import numpy as np

# Creating a DataFrame with the provided data
data = {
    'mobile_number': ['7000842753', '7001865778', '7001625959', '7001204172', '7000142493'],
    'circle_id': [109, 109, 109, 109, 109],
    'loc_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'std_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'loc_ic_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'last_date_of_month_6': ['6/30/2014', '6/30/2014', '6/30/2014', '6/30/2014', '6/30/2014'],
    'last_date_of_month_7': ['7/31/2014', '7/31/2014', '7/31/2014', '7/31/2014', '7/31/2014'],
    'last_date_of_month_8': ['8/31/2014', '8/31/2014', '8/31/2014', '8/31/2014', '8/31/2014'],
    'last_date_of_month_9': ['9/30/2014', '9/30/2014', '9/30/2014', '9/30/2014', '9/30/2014'],
    'arpu_6': [197.385, 34.047, 167.690, 221.338, 261.636],
    'sachet_3g_9': [0, 0, 0, 0, 0],
    'fb_user_6': [1.0, np.nan, np.nan, np.nan, 0.0],
    'fb_user_7': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_8': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_9': [np.nan, np.nan, 1.0, np.nan, np.nan],
    'aon': [968, 1006, 1103, 2491, 1526],
    'aug_vbc_3g': [30.4, 0.0, 0.0, 0.0, 0.0],
    'jul_vbc_3g': [0.0, 0.0, 0.0, 0.0, 0.0],
    'jun_vbc_3g': [101.20, 0.00, 4.17, 0.00, 0.00],
    'sep_vbc_3g': [3.58, 0.00, 0.00, 0.00, 0.00]
}

# Adding remaining columns to make the length 226
for i in range(20, 227):
    data[f'column_{i}'] = [np.nan] * 5

# Creating the DataFrame
df = pd.DataFrame(data)

# Displaying the DataFrame
print(df.head())


# In[10]:


import pandas as pd
import numpy as np

# Creating a DataFrame with the provided data
data = {
    'mobile_number': ['7000842753', '7001865778', '7001625959', '7001204172', '7000142493'],
    'circle_id': [109, 109, 109, 109, 109],
    'loc_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'std_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'loc_ic_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'last_date_of_month_6': ['6/30/2014', '6/30/2014', '6/30/2014', '6/30/2014', '6/30/2014'],
    'last_date_of_month_7': ['7/31/2014', '7/31/2014', '7/31/2014', '7/31/2014', '7/31/2014'],
    'last_date_of_month_8': ['8/31/2014', '8/31/2014', '8/31/2014', '8/31/2014', '8/31/2014'],
    'last_date_of_month_9': ['9/30/2014', '9/30/2014', '9/30/2014', '9/30/2014', '9/30/2014'],
    'arpu_6': [197.385, 34.047, 167.690, 221.338, 261.636],
    'sachet_3g_9': [0, 0, 0, 0, 0],
    'fb_user_6': [1.0, np.nan, np.nan, np.nan, 0.0],
    'fb_user_7': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_8': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_9': [np.nan, np.nan, 1.0, np.nan, np.nan],
    'aon': [968, 1006, 1103, 2491, 1526],
    'aug_vbc_3g': [30.4, 0.0, 0.0, 0.0, 0.0],
    'jul_vbc_3g': [0.0, 0.0, 0.0, 0.0, 0.0],
    'jun_vbc_3g': [101.20, 0.00, 4.17, 0.00, 0.00],
    'sep_vbc_3g': [3.58, 0.00, 0.00, 0.00, 0.00]
}

# Adding remaining columns to make the length 226
for i in range(20, 227):
    data[f'column_{i}'] = [np.nan] * 5

# Creating the DataFrame
df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
print(df.head())

# Drop rows with any NaN values
df_dropped_rows = df.dropna(axis=0)

# Displaying the DataFrame after dropping rows with any NaN values
print("\nDataFrame after dropping rows with any NaN values:")
print(df_dropped_rows)


# In[11]:


df.info()


# In[12]:


object_columns = df.select_dtypes(include=['object']).columns
print(object_columns)


# In[13]:


for col in object_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[14]:


df['mobile_number'] = pd.to_numeric(df['mobile_number'], errors='coerce')


# In[15]:


date_columns = [
    'last_date_of_month_6',
    'last_date_of_month_7',
    'last_date_of_month_8',
    'last_date_of_month_9'
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')


# In[16]:


df.describe()


# In[17]:


print(df.columns)


# In[18]:


import pandas as pd
import numpy as np

# Example DataFrame creation (use your actual DataFrame)
data = {
    'mobile_number': ['7000842753', '7001865778', '7001625959', '7001204172', '7000142493'],
    'circle_id': [109, 109, 109, 109, 109],
    'loc_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'std_og_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'loc_ic_t2o_mou': [0.0, 0.0, 0.0, 0.0, 0.0],
    'arpu_6': [197.385, 34.047, 167.690, 221.338, 261.636],
    'sachet_3g_9': [0, 0, 0, 0, 0],
    'fb_user_6': [1.0, np.nan, np.nan, np.nan, 0.0],
    'fb_user_7': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_8': [1.0, 1.0, np.nan, np.nan, np.nan],
    'fb_user_9': [np.nan, np.nan, 1.0, np.nan, np.nan],
    'aon': [968, 1006, 1103, 2491, 1526],
    'aug_vbc_3g': [30.4, 0.0, 0.0, 0.0, 0.0],
    'jul_vbc_3g': [0.0, 0.0, 0.0, 0.0, 0.0],
    'jun_vbc_3g': [101.20, 0.00, 4.17, 0.00, 0.00],
    'sep_vbc_3g': [3.58, 0.00, 0.00, 0.00, 0.00]
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Convert 'mobile_number' to string if not already
df['mobile_number'] = df['mobile_number'].astype(str)

# Convert other columns to numeric types
numeric_columns = ['circle_id', 'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou', 'arpu_6', 'sachet_3g_9', 'fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9', 'aon', 'aug_vbc_3g', 'jul_vbc_3g', 'jun_vbc_3g', 'sep_vbc_3g']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Verify DataFrame dtypes after cleaning
print(df.dtypes)
print(df.head())


# # Handling Missing Data

# In[19]:


import pandas as pd
import numpy as np

# Assuming df is your DataFrame

# Fill missing values in `fb_user_*` columns with 0
fb_user_columns = ['fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9']
df[fb_user_columns] = df[fb_user_columns].fillna(0)

# Optionally, fill missing values in other columns or use imputation
# For demonstration, filling all remaining NaNs with 0
df.fillna(0, inplace=True)

# Check for any remaining NaN values
print("Remaining NaN values:")
print(df.isna().sum())

# Summary statistics
print("Summary statistics:")
print(df.describe())

# Print the cleaned DataFrame
print(df.head())


# # Data Analysis

# In[20]:


# Example of a basic analysis: Calculate the mean ARPU
mean_arpu = df['arpu_6'].mean()
print(f"Mean ARPU: {mean_arpu}")

# Example: Calculate correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Example: Plot data (requires matplotlib)
import matplotlib.pyplot as plt

# Plot ARPU values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['arpu_6'], marker='o', linestyle='-')
plt.title('ARPU Over Records')
plt.xlabel('Index')
plt.ylabel('ARPU')
plt.grid(True)
plt.show()


# # Visualization:

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[22]:


# Histogram of ARPU
plt.figure(figsize=(10, 6))
plt.hist(df['arpu_6'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of ARPU')
plt.xlabel('ARPU')
plt.ylabel('Frequency')
plt.show()

# Scatter plot between `arpu_6` and `fb_user_6`
plt.figure(figsize=(10, 6))
plt.scatter(df['fb_user_6'], df['arpu_6'], color='green')
plt.title('ARPU vs. Facebook Usage')
plt.xlabel('Facebook User (6)')
plt.ylabel('ARPU')
plt.grid(True)
plt.show()


# # ##T-Test (for comparing two groups)

# In[23]:


from scipy.stats import ttest_ind

# Separate the data based on fb_user_6
fb_user_0 = df[df['fb_user_6'] == 0]['arpu_6'].dropna()
fb_user_1 = df[df['fb_user_6'] == 1]['arpu_6'].dropna()

# Perform t-test
t_stat, p_value = ttest_ind(fb_user_0, fb_user_1, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in ARPU based on Facebook usage.")
else:
    print("There is no significant difference in ARPU based on Facebook usage.")


# # ##ANOVA (for comparing more than two groups)

# In[24]:


from scipy.stats import f_oneway

# Prepare data for ANOVA
groups = [df[df['fb_user_6'] == value]['arpu_6'].dropna() for value in df['fb_user_6'].unique()]

# Perform ANOVA
f_stat, p_value = f_oneway(*groups)

print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in ARPU across different levels of Facebook usage.")
else:
    print("There is no significant difference in ARPU across different levels of Facebook usage.")


# In[25]:


##Post-hoc Analysis


# In[26]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Prepare data for Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df['arpu_6'].dropna(), groups=df['fb_user_6'].dropna(), alpha=0.05)

print(tukey)


# In[27]:


import matplotlib.pyplot as plt
import pandas as pd

# Example KPI data
data = {
    'Month': ['June', 'July', 'August', 'September'],
    'KPI': [6, 7, 8, 9]  # Replace these with actual KPI values
}

# Create a DataFrame
kpi_data = pd.DataFrame(data)

# Assuming KPI values are the ones to be plotted on y-axis and Months on x-axis (converted to numerical for hexbin plot)
kpi_data['Month_Num'] = kpi_data.index + 1

# Create hexbin plot
plt.hexbin(kpi_data['Month_Num'], kpi_data['KPI'], gridsize=20, cmap='Blues')
plt.colorbar(label='Counts')

# Set plot limits
plt.ylim(ymin=0)
plt.xlim(xmin=0)

# Add labels and title
plt.xlabel('Month')
plt.ylabel('KPI')
plt.title('Hexbin plot of Monthly KPI')

# Show plot
plt.show()


# In[28]:


import matplotlib.pyplot as plt
import pandas as pd

# Example data for ARPU and RECH
data = {
    'Month': ['June', 'July', 'August', 'September'],
    'ARPU': [10, 15, 20, 25],  # Replace these with actual ARPU values
    'RECH': [200, 250, 300, 350]  # Replace these with actual RECH values
}

# Create a DataFrame
kpi_data = pd.DataFrame(data)

# Create hexbin plot
plt.hexbin(kpi_data['ARPU'], kpi_data['RECH'], gridsize=20, cmap='Blues')
plt.colorbar(label='Counts')

# Set plot limits
plt.ylim(ymin=0)
plt.xlim(xmin=0)

# Add labels and title
plt.xlabel('Average Revenue Per User (ARPU)')
plt.ylabel('Recharge (RECH)')
plt.title('Hexbin plot of ARPU vs RECH')

# Show plot
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd

# Example data for ARPU and RECH
data = {
    'Month': ['June', 'July', 'August', 'September'],
    'ARPU': [10, 15, 20, 25],  # Replace these with actual ARPU values
    'RECH': [200, 250, 300, 350]  # Replace these with actual RECH values
}

# Create a DataFrame
kpi_data = pd.DataFrame(data)

# Calculate the maximum values for ARPU and RECH
max_arpu = kpi_data['ARPU'].max()
max_rech = kpi_data['RECH'].max()

print(f"Maximum ARPU: {max_arpu}")
print(f"Maximum RECH: {max_rech}")

# Create hexbin plot
plt.hexbin(kpi_data['ARPU'], kpi_data['RECH'], gridsize=20, cmap='Blues')
plt.colorbar(label='Counts')

# Set plot limits based on the data
plt.ylim(ymin=0, ymax=max_rech + 50)  # Adding some padding for better visualization
plt.xlim(xmin=0, xmax=max_arpu + 5)  # Adding some padding for better visualization

# Add labels and title
plt.xlabel('Average Revenue Per User (ARPU)')
plt.ylabel('Recharge (RECH)')
plt.title('Hexbin plot of ARPU vs RECH')

# Show plot
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import pandas as pd

# Example data for ARPU and RECH
data = {
    'Month': ['June', 'July', 'August', 'September'],
    'ARPU': [10, 15, 20, 25],  # Replace these with actual ARPU values
    'RECH': [200, 250, 300, 350]  # Replace these with actual RECH values
}

# Create a DataFrame
kpi_data = pd.DataFrame(data)

# Calculate the minimum and maximum values for ARPU and RECH
min_arpu = kpi_data['ARPU'].min()
max_arpu = kpi_data['ARPU'].max()
min_rech = kpi_data['RECH'].min()
max_rech = kpi_data['RECH'].max()

print(f"Minimum ARPU: {min_arpu}")
print(f"Maximum ARPU: {max_arpu}")
print(f"Minimum RECH: {min_rech}")
print(f"Maximum RECH: {max_rech}")

# Create hexbin plot
plt.hexbin(kpi_data['ARPU'], kpi_data['RECH'], gridsize=20, cmap='Blues')
plt.colorbar(label='Counts')

# Set plot limits based on the data
plt.ylim(ymin=min_rech - 50, ymax=max_rech + 50)  # Adding some padding for better visualization
plt.xlim(xmin=min_arpu - 5, xmax=max_arpu + 5)  # Adding some padding for better visualization

# Add labels and title
plt.xlabel('Average Revenue Per User (ARPU)')
plt.ylabel('Recharge (RECH)')
plt.title('Hexbin plot of ARPU vs RECH')

# Show plot
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import pandas as pd

# Example data for ARPU and RECH
data = {
    'Month': ['June', 'July', 'August', 'September'],
    'ARPU': [10, 15, 20, 25],  # Replace these with actual ARPU values
    'RECH': [200, 250, 300, 350]  # Replace these with actual RECH values
}

# Create a DataFrame
kpi_data = pd.DataFrame(data)

# Calculate the mean values for ARPU and RECH
mean_arpu = kpi_data['ARPU'].mean()
mean_rech = kpi_data['RECH'].mean()

print(f"Mean ARPU: {mean_arpu}")
print(f"Mean RECH: {mean_rech}")

# Create hexbin plot
plt.hexbin(kpi_data['ARPU'], kpi_data['RECH'], gridsize=20, cmap='Blues')
plt.colorbar(label='Counts')

# Set plot limits based on the data
plt.ylim(ymin=0, ymax=kpi_data['RECH'].max() + 50)  # Adding some padding for better visualization
plt.xlim(xmin=0, xmax=kpi_data['ARPU'].max() + 5)  # Adding some padding for better visualization

# Add mean lines
plt.axhline(y=mean_rech, color='red', linestyle='--', label=f'Mean RECH: {mean_rech:.2f}')
plt.axvline(x=mean_arpu, color='green', linestyle='--', label=f'Mean ARPU: {mean_arpu:.2f}')

# Add labels and title
plt.xlabel('Average Revenue Per User (ARPU)')
plt.ylabel('Recharge (RECH)')
plt.title('Hexbin plot of ARPU vs RECH')
plt.legend()

# Show plot
plt.show()


# In[32]:


df.columns


# In[33]:


df.describe()


# In[34]:


import pandas as pd

# Example DataFrame (replace this with your actual DataFrame)
data = {
    'mobile_number': [1, 2, 3, 4, 5],
    'circle_id': [101, 102, 103, 104, 105],
    'loc_og_t2o_mou': [10, 20, 30, 40, 50],
    'std_og_t2o_mou': [15, 25, 35, 45, 55],
    'loc_ic_t2o_mou': [5, 15, 25, 35, 45],
    'arpu_6': [100, 200, 300, 400, 500],
    'sachet_3g_9': [1, 2, 3, 4, 5],
    'fb_user_6': [1, 1, 0, 0, 1],
    'fb_user_7': [0, 1, 1, 1, 0],
    'fb_user_8': [1, 0, 1, 1, 1],
    'fb_user_9': [0, 1, 0, 1, 0],
    'aon': [100, 200, 300, 400, 500],
    'aug_vbc_3g': [50, 60, 70, 80, 90],
    'jul_vbc_3g': [20, 30, 40, 50, 60],
    'jun_vbc_3g': [10, 20, 30, 40, 50],
    'sep_vbc_3g': [60, 70, 80, 90, 100]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Calculate the average recharge amount for the first two months (June and July)
df['Average_Recharge'] = df[['jun_vbc_3g', 'jul_vbc_3g']].mean(axis=1)

# Calculate the 70th percentile of the average recharge amount
percentile_70 = df['Average_Recharge'].quantile(0.70)
print(f"70th percentile of average recharge amount: {percentile_70}")

# Filter high-value customers
high_value_customers = df[df['Average_Recharge'] >= percentile_70]

# Check the number of rows in the filtered dataset
print(f"Number of high-value customers: {high_value_customers.shape[0]}")

# Display the first few rows of the high-value customers dataset
print(high_value_customers.head())


# # Summary Statistics Breakdown
# #circle_id:
# 
# Count: 5
# Mean: 109
# Std Dev: 0 (all values are the same)
# Min, Max, Percentiles: All values are 109
# loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou:
# 
# #Count: 5
# Mean: 0 (indicating no usage or very low values)
# Std Dev: 0 (no variation in these columns)
# Min, Max, Percentiles: All values are 0
# arpu_6:
# 
# Mean: 176.42
# Std Dev: 86.70 (indicating variability in ARPU)
# Min: 34.05
# 25th Percentile: 167.69
# 50th Percentile (Median): 197.39
# 75th Percentile: 221.34
# Max: 261.64
# sachet_3g_9:
# 
# Count: 5
# Mean: 0
# Std Dev: 0 (all values are 0)
# Min, Max, Percentiles: All values are 0
# fb_user_6, fb_user_7, fb_user_8, fb_user_9:
# 
# Mean: Varies (0.20 to 0.40)
# Std Dev: Varies (0.45 to 0.55)
# Min: 0
# Max: 1
# These columns represent binary features (0 or 1), with varying proportions of 1s.
# aon:
# 
# Mean: 1418.80
# Std Dev: 639.23 (indicating significant variability)
# Min: 968
# 25th Percentile: 1006
# 50th Percentile (Median): 1103
# 75th Percentile: 1526
# Max: 2491
# aug_vbc_3g:
# 
# Mean: 6.08
# Std Dev: 13.60 (indicating some variability)
# Min: 0
# Max: 30.40
# jul_vbc_3g, jun_vbc_3g, sep_vbc_3g:
# 
# Mean: Varies (0.0 to 21.07)
# Std Dev: Varies (0.0 to 44.83)
# Min: 0
# Max: Varies (up to 101.20)
# Key Observations
# Constant Values: Columns like circle_id, loc_og_t2o_mou, std_og_t2o_mou, and sachet_3g_9 have constant values or very little variation, which might indicate missing data or columns that do not contribute much to analysis.
# High Variability: Columns like aon, arpu_6, and aug_vbc_3g show significant variability, indicating diverse data distributions.
# Binary Columns: fb_user_6, fb_user_7, fb_user_8, and fb_user_9 are binary features with varying frequencies of 1.

# In[35]:


# Calculate a lower percentile (e.g., 50th percentile)
percentile_50 = df['Average_Recharge'].quantile(0.50)
print(f"50th percentile of average recharge amount: {percentile_50}")

# Filter high-value customers using the new percentile
high_value_customers = df[df['Average_Recharge'] >= percentile_50]

# Check the number of rows in the filtered dataset
print(f"Number of high-value customers: {high_value_customers.shape[0]}")


# In[36]:


# Calculate the 40th percentile
percentile_40 = df['Average_Recharge'].quantile(0.40)
print(f"40th percentile of average recharge amount: {percentile_40}")

# Filter high-value customers using the 40th percentile
high_value_customers = df[df['Average_Recharge'] >= percentile_40]
print(f"Number of high-value customers: {high_value_customers.shape[0]}")


# In[37]:


# Plot distribution (if you have matplotlib installed)
import matplotlib.pyplot as plt

df['Average_Recharge'].hist(bins=20)
plt.title('Distribution of Average Recharge Amount')
plt.xlabel('Average Recharge')
plt.ylabel('Frequency')
plt.show()


# In[38]:


# Calculate the 30th percentile
percentile_30 = df['Average_Recharge'].quantile(0.30)
print(f"30th percentile of average recharge amount: {percentile_30}")

# Filter high-value customers using the 30th percentile
high_value_customers = df[df['Average_Recharge'] >= percentile_30]
print(f"Number of high-value customers: {high_value_customers.shape[0]}")


# In[39]:


# Summary statistics
print(df['Average_Recharge'].describe())

# Plot distribution
import matplotlib.pyplot as plt

df['Average_Recharge'].hist(bins=30)
plt.title('Distribution of Average Recharge Amount')
plt.xlabel('Average Recharge')
plt.ylabel('Frequency')
plt.show()


# In[40]:


# Summary statistics
print(df['Average_Recharge'].describe())

# Plot distribution
import matplotlib.pyplot as plt

df['Average_Recharge'].hist(bins=30)
plt.title('Distribution of Average Recharge Amount')
plt.xlabel('Average Recharge')
plt.ylabel('Frequency')
plt.show()


# # Based on the summary statistics of the Average_Recharge column:
# 
# Count: 5
# Mean: 35.00
# Standard Deviation: 15.81
# Minimum: 15.00
# 25th Percentile: 25.00
# 50th Percentile (Median): 35.00
# 75th Percentile: 45.00
# Maximum: 55.00

# In[41]:


import matplotlib.pyplot as plt

# Plot distribution of Average_Recharge
plt.hist(df['Average_Recharge'], bins=5, edgecolor='black')
plt.title('Distribution of Average Recharge Amount')
plt.xlabel('Average Recharge')
plt.ylabel('Frequency')
plt.show()


# In[42]:


# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Describe the data again after cleaning
print(df.describe())


# In[43]:


import matplotlib.pyplot as plt

# Plot the distribution of Average_Recharge
plt.figure(figsize=(10, 6))
plt.hist(df['Average_Recharge'], bins=10, edgecolor='black')
plt.title('Distribution of Average Recharge Amount')
plt.xlabel('Average Recharge')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[44]:


# Box plot of Average_Recharge
plt.figure(figsize=(10, 6))
plt.boxplot(df['Average_Recharge'])
plt.title('Box Plot of Average Recharge Amount')
plt.ylabel('Average Recharge')
plt.grid(True)
plt.show()


# In[45]:


import matplotlib.pyplot as plt

# Plot histograms of key features
features = ['Average_Recharge', 'arpu_6', 'aon', 'aug_vbc_3g', 'jul_vbc_3g', 'jun_vbc_3g', 'sep_vbc_3g']

plt.figure(figsize=(14, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[feature], bins=5, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[46]:


import pandas as pd

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'mobile_number': [1, 2, 3, 4, 5],
    'total_ic_mou_9': [0, 10, 0, 0, 0],
    'total_og_mou_9': [0, 20, 0, 0, 0],
    'vol_2g_mb_9': [0, 5, 0, 0, 0],
    'vol_3g_mb_9': [0, 10, 0, 0, 0],
    'other_feature': [100, 200, 300, 400, 500]  # Example non-churn phase feature
}

df = pd.DataFrame(data)

# Tag churners: customers who have not made any calls or used mobile internet in the churn phase
df['churn'] = ((df['total_ic_mou_9'] == 0) &
               (df['total_og_mou_9'] == 0) &
               (df['vol_2g_mb_9'] == 0) &
               (df['vol_3g_mb_9'] == 0)).astype(int)

# Display the DataFrame with churn tags
print("DataFrame with churn tags:")
print(df)

# Remove churn phase attributes (all columns with '_9')
columns_to_keep = [col for col in df.columns if '_9' not in col]
df_cleaned = df[columns_to_keep]

# Display the cleaned DataFrame
print("\nDataFrame after removing churn phase attributes:")
print(df_cleaned)


# # Churn Tags:
# 
# Customers are correctly tagged as churned (1) or not (0) based on the churn phase attributes. For example, customers with mobile_number 1, 3, 4, and 5 are tagged as churned because they didn't make any calls or use mobile internet in the churn phase.
# Cleaned DataFrame:
# 
# The cleaned DataFrame has had all churn phase attributes removed, leaving only relevant features like mobile_number, other_feature, and churn.

# In[47]:


df_cleaned.to_csv('cleaned_data.csv', index=False)


# # Exploratory Data Analysis (EDA):

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned DataFrame (replace 'cleaned_data.csv' with your actual file)
df_cleaned = pd.read_csv('cleaned_data.csv')


# In[49]:


# Summary Statistics
print("Summary Statistics:")
print(df_cleaned.describe(include='all'))


# In[50]:


# Distribution of Churned vs. Non-Churned Customers
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=df_cleaned)
plt.title('Distribution of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()


# In[51]:


# Compute proportion of churned vs. non-churned customers
churn_proportion = df_cleaned['churn'].value_counts(normalize=True)
print("\nProportion of Churned vs. Non-Churned Customers:")
print(churn_proportion)


# In[52]:


# Feature Analysis
features = df_cleaned.columns[df_cleaned.columns != 'churn']

for feature in features:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_cleaned, x=feature, hue='churn', multiple='stack')
    plt.title(f'Distribution of {feature} by Churn')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()


# In[53]:


# Correlation Analysis
correlation_matrix = df_cleaned.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[54]:


# Pair Plot
sns.pairplot(df_cleaned, hue='churn', diag_kind='kde')
plt.title('Pair Plot of Features by Churn')
plt.show()


# In[55]:


import matplotlib.pyplot as plt

# Example: After creating a plot
plt.tight_layout()
plt.show()


# In[56]:


plt.figure(figsize=(12, 8))  # Adjust the size as needed


# In[57]:


import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


# In[58]:


pip install --upgrade seaborn matplotlib


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned DataFrame (replace 'cleaned_data.csv' with your actual file)
df_cleaned = pd.read_csv('cleaned_data.csv')

# Distribution of Churned vs. Non-Churned Customers
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=df_cleaned)
plt.title('Distribution of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.tight_layout()  # Adjust layout
plt.show()


# In[60]:


pip install --upgrade seaborn matplotlib


# In[61]:


pip install --upgrade seaborn matplotlib --user


# In[62]:


pip install seaborn matplotlib


# # Feature Engineering:

# In[ ]:


import pandas as pd
df=pd.read_csv('C:\Users\HP\Downloads\cleaned_data.csv')


# In[ ]:


df = pd.read_csv(r'C:\Users\HP\Downloads\cleaned_data.csv')


# In[ ]:


# Check the first few rows of the DataFrame
print(df.head())


# In[ ]:


import pandas as pd


# Check for missing values
print(df.isnull().sum())


# In[ ]:


df_cleaned = df.dropna()  # Drop rows with missing values
# or
df_cleaned = df.dropna(axis=1)  # Drop columns with missing values


# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with 'median', 'most_frequent', or other strategies
df_imputed = pd.DataFrame(imputer.fit_transform(df))


# In[ ]:


### seveth august work


# In[64]:


# Columns for recharge amounts in the first two months (June and July)
recharge_columns = ['RECH_AMT.6', 'RECH_AMT.7']

# Calculate the average recharge amount for the first two months
data['avg_recharge_amt'] = data[recharge_columns].mean(axis=1)

# Calculate the 70th percentile of the average recharge amount
percentile_70 = np.percentile(data['avg_recharge_amt'], 70)

# Filter high-value customers
high_value_customers = data[data['avg_recharge_amt'] >= percentile_70]

# Verify the number of rows after filtering
print(f"Number of high-value customers: {high_value_customers.shape[0]}")

# Saving the filtered data to a new CSV file for further analysis
high_value_customers.to_csv('high_value_customers.csv', index=False)

# Display the first few rows of the filtered data
high_value_customers.head()


# In[65]:


# Columns for recharge amounts in the first two months (June and July)
# Update the column names based on the correct names in your dataset
recharge_columns = ['RECH_AMT.6', 'RECH_AMT.7']

# Ensure that the recharge columns exist in the DataFrame
for col in recharge_columns:
    if col not in data.columns:
        raise ValueError(f"Column {col} is not in the DataFrame")

# Calculate the average recharge amount for the first two months
data['avg_recharge_amt'] = data[recharge_columns].mean(axis=1)

# Calculate the 70th percentile of the average recharge amount
percentile_70 = np.percentile(data['avg_recharge_amt'], 70)

# Filter high-value customers
high_value_customers = data[data['avg_recharge_amt'] >= percentile_70]

# Verify the number of rows after filtering
print(f"Number of high-value customers: {high_value_customers.shape[0]}")

# Saving the filtered data to a new CSV file for further analysis
high_value_customers.to_csv('high_value_customers.csv', index=False)

# Display the first few rows of the filtered data
print(high_value_customers.head())


# In[67]:


df


# In[ ]:





# In[ ]:





# In[ ]:




