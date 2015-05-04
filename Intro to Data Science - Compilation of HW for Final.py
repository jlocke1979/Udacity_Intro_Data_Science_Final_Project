
# coding: utf-8

# # Section 1 - Mann-Whitney Test

# In[5]:

import numpy as np
import scipy
import scipy.stats
import pandas

def mann_whitney_plus_means(turnstile_weather):
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data. 
    
    You will want to take the means and run the Mann Whitney U-test on the 
    ENTRIESn_hourly column in the turnstile_weather dataframe.
    
    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain
    
    You should feel free to use scipy's Mann-Whitney implementation, and you 
    might also find it useful to use numpy's mean function.
    
    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    
    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    ### YOUR CODE HERE ###
    #filter data
    with_rain_turnstile_weather = turnstile_weather[turnstile_weather['rain'] == 1]
    without_rain_turnstile_weather = turnstile_weather[turnstile_weather['rain'] == 0]
    #Calculate N1 and N2
    n1 = len(with_rain_turnstile_weather)
    n2 = len(without_rain_turnstile_weather)
    print n1
    print n2
    #Calculate mu
    mu = (n1*n2)/2
    print mu
    #Calculate ou (sigma
    ou = ((n1*n2*(n1+n2+1))/12)**(.5)
    print ou
    # calculate Z
    U = 1924409167.0
    z = (U - mu) / ou
    print z
    # calculate p
    pval = 2*scipy.stats.norm.cdf(z)
    print pval
    #Calculate Means
    with_rain_mean = np.mean(with_rain_turnstile_weather['ENTRIESn_hourly'])
    without_rain_mean = np.mean(without_rain_turnstile_weather['ENTRIESn_hourly'])
    
    #Calculate MannWhitney Stat 
    x = with_rain_turnstile_weather['ENTRIESn_hourly']
    y = without_rain_turnstile_weather['ENTRIESn_hourly']
    U,p = scipy.stats.mannwhitneyu(x,y)
    print p*2
    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader

   
def convert_to_dataframe(file_name):
    df = pandas.read_csv(file_name)
    return df

    
file_location = 'C:\\Users\\Justin\\Dropbox\\Education\\Udacity - Data Analyst\\Intro to Data Science\\Lesson 3\\HW\\Q3 - Data\\turnstile_data_master_with_weather.csv'
df_turnstile_data = convert_to_dataframe(file_location)

mann_whitney_plus_means(df_turnstile_data)


# # Section 2- Linear Regression using Statsmodels

# In[9]:

import numpy as np
import pandas
import scipy
import statsmodels.api as sm

    
def predictions(weather_turnstile):
    
    # Select Features (try different features!)
    features = weather_turnstile[['Hour','maxpressurei','maxdewpti','mindewpti','minpressurei','meanpressurei','meanwindspdi','mintempi']]

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    # Add Constanct
    features = sm.add_constant(features)
    values = weather_turnstile['ENTRIESn_hourly']

    model = sm.OLS(values,features)
    results = model.fit()
    ############## PRINT THESE TO GET RESULTS#############
    #print results.params  # Print for Co-efficients
    #print results.tvalues  # Print for t-Values
    print results.summary()
    ####FOR AUTOGRADER, FINAL PREDICTION##################
    prediction = results.predict(features)
    return prediction
   
def convert_to_dataframe(file_name):
    df = pandas.read_csv(file_name)
    return df

file_location = 'C:\\Users\\Justin\\Dropbox\\Education\\Udacity - Data Analyst\\Intro to Data Science\\Lesson 3\\HW\\Q3 - Data\\turnstile_data_master_with_weather.csv'
df_turnstile_data = convert_to_dataframe(file_location)

predictions(df_turnstile_data)
   


# # Section 3a - Histogram

# In[10]:

import numpy as np
import pandas
import matplotlib.pyplot as plt

def entries_histogram(turnstile_weather):

    df_turnstile_weather = pandas.read_csv(turnstile_weather)
    plt.figure
    plt.axis([0, 55000, 0, 55000]) 
    binBoundaries = np.linspace(0,55000,100)
    
    # filter between rain and no Rain dataset
    no_rain_turnstile_weather = df_turnstile_weather[df_turnstile_weather.rain == 0]
    rain_turnstile_weather = df_turnstile_weather[df_turnstile_weather.rain == 1]
    
    # Plot a historgram for hourly entries when it is not raining
    no_rain_turnstile_weather['ENTRIESn_hourly'].hist(bins=binBoundaries, label='No Rain') 
    # Plot a historgram for hourly entries when it is raining
    rain_turnstile_weather['ENTRIESn_hourly'].hist(bins=binBoundaries, label='Rain') 
    #plt.axvline(no_rain_turnstile_weather['ENTRIESn_hourly'].mean(), color='b', linestyle='dashed', linewidth=2)
    #plt.axvline(rain_turnstile_weather['ENTRIESn_hourly'].mean(), color='b', linestyle='dashed', linewidth=2)
    plt.title('Histogram of Entries in No Rain vs Rain ')
    plt.xlabel('Number of Entries')
    plt.ylabel('Frequency of observations in each bin')
    plt.legend()    
    return plt

file_location = 'C:\\Users\\Justin\\Dropbox\\Education\\Udacity - Data Analyst\\Intro to Data Science\\Lesson 3\\HW\\Q1 - Data\\turnstile_data_master_with_weather.csv'
pic = entries_histogram(file_location)

plt.show(pic) 


# # Section 3b - By Hour of Day

# In[11]:

# Works in Autograder and MY computer
from pandas import *
from ggplot import *
import datetime

def plot_weather_data(turnstile_weather):
    grouped = turnstile_weather.groupby(['Hour'], sort=True,as_index=False).sum()
    plot = ggplot(grouped,aes(x='Hour', y='ENTRIESn_hourly')) +     geom_line(color = 'blue') +     ggtitle('Ridership by Hour of Day') +     ylab('Number of Entries') +     xlab('Hour of the day') +     scale_x_continuous(minor_breaks =(1,3,5,7,9,11,13,15,17,19,21,23), breaks=(0,2,4,6,8,10,12,14,16,18,20,22))
    return plot

def convert_to_dataframe(file_name):
    df = pandas.read_csv(file_name)
    return df

file_location = 'C:\\Users\\Justin\\Dropbox\\Education\Udacity - Data Analyst\\Intro to Data Science\\Lesson 3\\HW\\Q3 - Data\\turnstile_data_master_with_weather.csv'
df = convert_to_dataframe(file_location)
#print df

plot_weather_data(df)


# In[ ]:



