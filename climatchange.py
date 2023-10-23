# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author: Angelos Alamanos
"""

import os
import pandas as pd

# Define the path to your CSV file with the metorological data (change as needed)
file_path = r'D:\your\path\to\inputdata.csv'

data = pd.read_csv(file_path, sep=',')
print(data)

# Replace 'M' with NaN
data.replace('M', float('nan'), inplace=True)

# Convert columns to float
data[['Precipitation', 'Snowfall', 'Avg. Max Temp', 'Avg. Min Temp', 'Avg. Mean Temp']] = data[['Precipitation', 'Snowfall', 'Avg. Max Temp', 'Avg. Min Temp', 'Avg. Mean Temp']].astype(float)

# Define a function to fill missing values with the mean of the respective month
def fill_missing_with_monthly_mean(column_name):
    data[column_name] = data.groupby(data['Date'].str.extract(r'(\w+)-\d+')[0])[column_name].transform(lambda x: x.fillna(x.mean()))

# Fill Missing Values with the mean of the respective months
for column in ['Precipitation', 'Snowfall', 'Avg. Max Temp', 'Avg. Min Temp', 'Avg. Mean Temp']:
    fill_missing_with_monthly_mean(column)

# Specify the file path for the new CSV file
output_file_path = r"D:\your\path\to\timeseries_filled_values.csv"

# Save the updated data to the new file with the same separator '\t'
data.to_csv(output_file_path, sep=',', index=False)
print(f"Updated data saved to {output_file_path}")


# Validation
# Check for missing values
missing_values = data.isnull().sum()

# Print summary statistics
summary_statistics = data.describe()

print("Missing Values:")
print(missing_values)
print("\nSummary Statistics:")
print(summary_statistics)


####################### Convert Data Units  #######################

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your original CSV file and the path for the new CSV file (change as needed)
original_csv_file = r'D:\your\path\to\timeseries_filled_values.csv'
new_csv_file = r'D:\your\path\to\timeseries_converted_values.csv'

# Read the data from the original CSV file
df = pd.read_csv(original_csv_file, sep=',')

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Convert Precipitation from inches to mm (1 inch = 25.4 mm)
df['Precipitation'] = df['Precipitation'] * 25.4

# Convert Temperature from Fahrenheit to Celsius (°C = (°F - 32) / 1.8)
df['Avg. Max Temp'] = (df['Avg. Max Temp'] - 32) / 1.8
df['Avg. Min Temp'] = (df['Avg. Min Temp'] - 32) / 1.8
df['Avg. Mean Temp'] = (df['Avg. Mean Temp'] - 32) / 1.8

# Save the updated data to a new CSV file
df.to_csv(new_csv_file, sep=',', index=True)


######################    Time Series Plots    ###############################

# Change the path to your data!
data_file = r'D:\your\path\to\timeseries_converted_values.csv'

# Read the data from the CSV file
df = pd.read_csv(data_file, sep=',')

# Convert the 'Date' column to a datetime format  (if necessary - some steps might not apply to your data, so check carefully)
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Read the data from the CSV file
df = pd.read_csv(data_file, sep=',')

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Create Time Series Plots (note that the data column names are always specific to the input file)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Precipitation'], label='Precipitation', color='blue')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df.index, df['Avg. Mean Temp'], label='Avg. Mean Temp', color='red')
plt.xlabel('Year')
plt.ylabel('Temperature (oC)')
plt.legend()

plt.tight_layout()


# Create a Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Create 3 Box Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.boxplot(data=df, y='Precipitation', showfliers=False)
plt.ylabel('Precipitation (mm)')

plt.subplot(1, 3, 2)
sns.boxplot(data=df, y='Avg. Max Temp', showfliers=False)
plt.ylabel('Temperature (oC)')

plt.subplot(1, 3, 3)
sns.boxplot(data=df, y='Avg. Min Temp', showfliers=False)
plt.ylabel('Temperature (oC)')

# Show the plots
plt.show()


####  Monthly Box Plots # (note that the data column names are always specific to the input file)

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your CSV file (change)
csv_file = r'D:\your\path\to\timeseries_converted_values.csv'

# Read the data from the CSV file
df = pd.read_csv(csv_file, sep=',')

# Create a new column 'Date' by parsing the 'Date' column
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Extract month and year from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.strftime('%b')  # Convert month to a three-letter abbreviation

# Create box plots for Mean Precipitation and Temperature (monthly) with custom colors
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Month', y='Precipitation', showfliers=False, color='lightblue')
plt.ylabel('Precipitation (mm)')
plt.xlabel('Month')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Month', y='Avg. Mean Temp', showfliers=False, color='lightcoral')
plt.ylabel('Temperature (oC)')
plt.xlabel('Month')
plt.xticks(rotation=45)

# Show the plots
plt.tight_layout()
plt.show()


###############################################################################
###############    Downloading Climate Change projections    ##################

########################    RCP 4.5 ################################

# ## Import required packages

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import re
import urllib.request
import numpy as np
from datetime import datetime, timedelta


# ## Define GCM cell close to location (e.g. assume our location has these coordinates and is called "Creek")

def find_nearest_cell(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

latarray = np.linspace(-89.875,89.875,720)
lonarray = np.linspace(0.125,359.875,1440)
Creek = (41.3366, 360-85.1198) # note that longitude must be on degrees east!!

celllatindex = find_nearest_cell(latarray,Creek[0])
celllonindex = find_nearest_cell(lonarray,Creek[1])
print(celllatindex,celllonindex)  # this returns the cell of our area


########### How to find your start and end days  ############

# Define the start and end dates
start_date = datetime(1850, 1, 1)
end_date = datetime(2050, 1, 1) # change this date with your preferred one!!

# Calculate the difference between the two dates
time_difference = end_date - start_date

# Extract the number of days from the time difference
number_of_days = time_difference.days

print("Number of days:", number_of_days)
################################################

# ## Define begin and end date of GCM simulation

zerodate = datetime(1850,1,1)
zerodate.isoformat(' ')

begindate = zerodate + timedelta(days=56978.5)   # this is 2006,1,1
begindate.isoformat(' ')

enddate = zerodate + timedelta(days=73049)# this is 2050,1,1
enddate.isoformat(' ')


# ## Get data from the NCCS server by 5000 records 

# We divide the request by groups of 5000 records because the server does not provide all the records.
intervals=[[0,4999],[5000,9999],[10000,14999],[15000,19999],[20000,24999],[25000,29999],[30000,34674]]

#empty array for ppt or T and time
pptlist = []
daylist = []

# Define the regular expression patterns for time and precipitation
time_pattern = re.compile(r"pr\.time = \{(\d+), (\d+)\};")
ppt_pattern = re.compile(r"pr = \{(\d+), (\d+)\};")

# Search for time and precipitation data in the response text
time_match = time_pattern.search(mystr)
ppt_match = ppt_pattern.search(mystr)

if time_match and ppt_match:
    time_start = int(time_match.group(1))
    time_end = int(time_match.group(2))
    ppt_start = int(ppt_match.group(1))
    ppt_end = int(ppt_match.group(2))

    time_data = list(map(int, lines[time_start:time_end + 1].split(',')))
    ppt_data = list(map(float, lines[ppt_start:ppt_end + 1].split(',')))

    # Calculate the date for the first day (assuming zerodate is 1850-01-01)
    current_date = zerodate + timedelta(days=time_data[0])

    for day, ppt in zip(time_data, ppt_data):
        daylist.append(current_date)
        pptlist.append(ppt)
        current_date += timedelta(days=1)

    
for interval in intervals:
    fp = urllib.request.urlopen("https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/NEX-GDDP/bcsd/rcp45/r1i1p1/pr/CSIRO-Mk3-6-0.ncml.ascii?pr["+str(interval[0])+":1:"+str(interval[1])+"]["+str(celllatindex)+":1:"+str(celllatindex)+"]["+str(celllonindex)+":1:"+str(celllonindex)+"]")

# In case of Historic Data, from 1949 to 2005
#    fp = urllib.request.urlopen("https://dataserver.nccs.nasa.gov/thredds/dodsC/\
#bypass/NEX-GDDP/bcsd/historical/r1i1p1/pr/CSIRO-Mk3-6-0.ncml.ascii?pr\
#["+str(interval[0])+":1:"+str(interval[1])+"]\
#["+str(celllatindex)+":1:"+str(celllatindex)+"]\
#["+str(celllonindex)+":1:"+str(celllonindex)+"]")
    
    mybytes = fp.read()

    mystr = mybytes.decode("utf8")
    fp.close()
    
    lines = mystr.split('\n')
    breakers = []
    breakerTexts = ['pr[time','pr.pr','pr.time']
    for line in lines:
        for text in breakerTexts:
            if text in line:
                breakers.append(lines.index(line))
                
    dayline = lines[breakers[0]]
    dayline = re.sub('\[|\]',' ',dayline)
    days = int(dayline.split()[4])
    print("Procesing interval %s of %d days" % (str(interval), days))
    
    for item in range(breakers[1]+1, breakers[1]+days+1):
        ppt = float(lines[item].split(',')[1])*86400
        pptlist.append(ppt)
        
    for day in lines[breakers[2]+1].split(','):
        daylist.append(zerodate + timedelta(days=float(day)))

plt.plot(daylist,pptlist)
plt.gcf().autofmt_xdate()
plt.ylabel('Precipitation (mm/day)')
plt.show()


# Save the data as a CSV file
import pandas as pd

# Create a DataFrame from the collected data
data = {'Date': daylist, 'Precipitation (mm/day)': pptlist}
df = pd.DataFrame(data)

# Define the path to save the CSV file
output_path = r'D:\your path to \downloaded future projections\p_rcp45.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_path, index=False)

print("Data saved to:", output_path)



######################  READ THE CLIMATE CHANGE DAILY PROJECTIONS #########################
#############  CONVERT THEM TO MONTHLY DATA, IN THE SAME FORMAT WITH THE HISTORIC DATA ######
#############  SAVE THE MONTHLY RESULTS IN THE FOLDER & PLOT THE RESULTS  ##################

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the input and output paths
input_path = r"D:\your path to downloaded future projections\... .csv"
output_path = r"D:\your path to downloaded future projections\...monthly.csv"
output_plot_path = r"D:\your path to downloaded future projections\...monthly_boxplots.png"

# Read the daily climate change data and convert the 'Date' column to datetime
climate_data = pd.read_csv(input_path)
climate_data['Date'] = pd.to_datetime(climate_data['Date'])

# Convert daily data to monthly data
monthly_data = climate_data.resample('M', on='Date').sum()

# Format the date column as Month-Year
monthly_data['Month-Year'] = monthly_data.index.strftime('%b-%Y')

# Save the monthly data with the desired format
monthly_data.to_csv(output_path, columns=['Month-Year', 'Precipitation (mm/day)'], index=False, header=['Date', 'Precipitation'])


# Plot the monthly data over time (years)
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Precipitation (mm/day)'])
plt.xlabel('Date')
plt.ylabel('Monthly Precipitation (mm)')
plt.title('Monthly Precipitation (RCP 8.5) Over Time')
plt.grid(True)
plt.savefig(output_plot_path)
plt.show()

# Create monthly box plots

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

monthly_data['Month'] = pd.Categorical(monthly_data.index.strftime('%b'), categories=month_order, ordered=True)
boxplot = monthly_data.boxplot(column='Precipitation (mm/day)', by='Month', showfliers=False)
plt.title('Monthly Precipitation (RCP 8.5) Box Plots')
plt.suptitle('')
plt.xlabel('Month')
plt.ylabel('Monthly Precipitation (mm)')
plt.grid(True)
plt.savefig(output_plot_path)
plt.show()




###############################################################################
#######################       BIAS CORRECTION       ###########################

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Load historical and projected data
hist_data = pd.read_csv("D:/your path to historic data/... hist.csv") # In the examples below we use indicatively the name Precipitation_hist
proj_data = pd.read_csv("D:/your path to projected data, as converted into monthly/... proj.csv") # In the examples below we use indicatively the name Precipitation_proj

# Ensure the date format is consistent
hist_data['Date'] = pd.to_datetime(hist_data['Date'], format="%b-%y")
proj_data['Date'] = pd.to_datetime(proj_data['Date'], format="%b-%y")

# Filter data for the common period (2005-2022) # change as per your data
common_period_start = pd.to_datetime("Nov-05", format="%b-%y")
common_period_end = pd.to_datetime("Oct-22", format="%b-%y")
hist_data = hist_data[(hist_data['Date'] >= common_period_start) & (hist_data['Date'] <= common_period_end)]
proj_data = proj_data[(proj_data['Date'] >= common_period_start) & (proj_data['Date'] <= common_period_end)]

# Merge data on the date field
merged_data = pd.merge(hist_data, proj_data, on='Date', suffixes=('_hist', '_proj'))


# Run this part of the script, and then run one of the alternative methods provided below.


###################   DELTA TEST (Change) Method #######################
# Calculate monthly deltas
deltas = merged_data['Precipitation_proj'] - merged_data['Precipitation_hist']

# Calculate the average monthly delta change
average_delta_change = np.mean(deltas)

# Apply bias correction by subtracting the average delta change
corrected_proj_data = merged_data.copy()
corrected_proj_data['Precipitation_proj'] = merged_data['Precipitation_proj'] - average_delta_change

# Calculate R-squared (R²)
r_squared = r2_score(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj'])

# Calculate Relative Error (RE)
relative_error = np.mean(abs(corrected_proj_data['Precipitation_proj'] - merged_data['Precipitation_hist']) / merged_data['Precipitation_hist'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj']))

# Print the results
print(f"R-squared (R²): {r_squared}")
print(f"Relative Error (RE): {relative_error}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save corrected data to a CSV file (change as preferred)
corrected_proj_data.to_csv("D:/your path to the output file/... .csv", index=False)


###################   Multiplicative Scaling Method #######################

# Calculate the scaling factor
scaling_factor = np.mean(merged_data['Precipitation_hist']) / np.mean(merged_data['Precipitation_proj'])

# Apply bias correction using the scaling factor
corrected_proj_data = merged_data.copy()
corrected_proj_data['Precipitation_proj'] = merged_data['Precipitation_proj'] * scaling_factor

# Calculate R-squared (R²)
r_squared = r2_score(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj'])

# Calculate Relative Error (RE)
relative_error = np.mean(abs(corrected_proj_data['Precipitation_proj'] - merged_data['Precipitation_hist']) / merged_data['Precipitation_hist'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj']))

# Print the results
print(f"R-squared (R²): {r_squared}")
print(f"Relative Error (RE): {relative_error}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save corrected data to a CSV file (change as preferred)
corrected_proj_data.to_csv("D:/your path to the output file/... .csv", index=False)


###################   Quantile Mapping Method #######################

# Calculate quantiles for historical and projected data
hist_quantiles = np.arange(0, 1.05, 0.05)  # 21 quantiles
proj_quantiles = np.percentile(merged_data['Precipitation_proj'], hist_quantiles * 100)

# Map projected data to historical data quantiles
quantile_mapping = dict(zip(proj_quantiles, hist_quantiles))
corrected_proj_data = merged_data.copy()
corrected_proj_data['Precipitation_proj'] = merged_data['Precipitation_proj'].map(quantile_mapping)

# Calculate R-squared (R²)
r_squared = r2_score(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj'])

# Calculate Relative Error (RE)
relative_error = np.mean(abs(corrected_proj_data['Precipitation_proj'] - merged_data['Precipitation_hist']) / merged_data['Precipitation_hist'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj']))

# Print the results
print(f"R-squared (R²): {r_squared}")
print(f"Relative Error (RE): {relative_error}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save corrected data to a CSV file
corrected_proj_data.to_csv("D:/your path to the output file/... .csv", index=False)


###################   CF Method #######################

# Calculate CDF for historical and projected data
hist_data['CDF_hist'] = hist_data['Precipitation'].rank() / len(hist_data)
proj_data['CDF_proj'] = proj_data['Precipitation'].rank() / len(proj_data)

# Merge data with CDF values
merged_data = pd.merge(merged_data, hist_data[['Date', 'CDF_hist']], on='Date')
merged_data = pd.merge(merged_data, proj_data[['Date', 'CDF_proj']], on='Date')

# Apply bias correction using CDF matching
corrected_proj_data = merged_data.copy()
corrected_proj_data['Precipitation_proj'] = merged_data['Precipitation_hist'].quantile(merged_data['CDF_proj'])

# Calculate R-squared (R²)
r_squared = r2_score(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj'])

# Calculate Relative Error (RE)
relative_error = np.mean(abs(corrected_proj_data['Precipitation_proj'] - merged_data['Precipitation_hist']) / merged_data['Precipitation_hist'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(merged_data['Precipitation_hist'], corrected_proj_data['Precipitation_proj']))

# Print the results
print(f"R-squared (R²): {r_squared}")
print(f"Relative Error (RE): {relative_error}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save corrected data to a CSV file
corrected_proj_data.to_csv("D:/your path to the output file/... .csv", index=False)


