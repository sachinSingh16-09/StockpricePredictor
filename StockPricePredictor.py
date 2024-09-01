#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas, numpy and matplotlib.pyplot


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# read the csv file containing dataset
data = pd.read_csv('GOOG_30min_sample.csv')


# In[4]:


# display the dataset
data


# In[5]:


# plot a graph showing the stock's closing price for each timestamp

plt.figure(figsize = (16, 6))
plt.title('Close Price History')
plt.plot(data['close'])
plt.xlabel('timestamp', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()


# In[6]:


# filter out the training data from the dataset
# here 95 percent of the dataset is taken as the training data

df = data.filter(['close'])
dataset = df.values
training_data_len = int(np.ceil(len(dataset) * .95))

training_data_len


# In[7]:


# normalize the data to a range between 0 and 1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[8]:


# create a new training dataset "train_data" that contains the scaled_data

train_data = scaled_data[0:int(training_data_len), :]

x_train = []   
y_train = []

# Split the data into x_train and y_train data sets

for i in range(31, len(train_data)):
    x_train.append(train_data[i-31:i , 0])
    y_train.append(train_data[i, 0])
    if i <= 32:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays 

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# x_train.shape    


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#train the model upto 10 epochs(depending upon the loss)

model.fit(x_train, y_train, batch_size= 1, epochs = 10)


# In[10]:


#create a testing data set
test_data = scaled_data[training_data_len - 31:, :]

#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(31, len(test_data)):
    x_test.append(test_data[i-31:i, 0])

#convert the data into a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#store the model's predicted prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions- y_test) ** 2)))
rmse
#root mean square error(rmse)


# In[11]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predicted close'] = predictions

plt.figure(figsize = (16,6))
plt.title('Model')
plt.xlabel('Date', fontsize =18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predicted close']])
plt.legend(['Train', 'val', 'Predicted close'], loc='lower right')
plt.show()


# In[12]:


#show the prediced closing value
valid


# In[14]:


# Create DataFrame
df = pd.DataFrame(valid)

# Convert 'timestamp' to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Calculate short-term and long-term moving averages
short_window = 3  # Adjust as needed
long_window = 6   # Adjust as needed

df['Short_MA'] = df['Predicted close'].rolling(window=short_window).mean()
df['Long_MA'] = df['Predicted close'].rolling(window=long_window).mean()

# Initialize signals
df['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell

# Generate trading signals
for i in range(1, len(df)):
    if not pd.isna(df['Short_MA'].iloc[i]) and not pd.isna(df['Long_MA'].iloc[i]) and \
       df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i] and \
       df['Short_MA'].iloc[i-1] <= df['Long_MA'].iloc[i-1]:
        df.at[df.index[i], 'Signal'] = 1  # Buy
    elif not pd.isna(df['Short_MA'].iloc[i]) and not pd.isna(df['Long_MA'].iloc[i]) and \
         df['Short_MA'].iloc[i] < df['Long_MA'].iloc[i] and \
         df['Short_MA'].iloc[i-1] >= df['Long_MA'].iloc[i-1]:
        df.at[df.index[i], 'Signal'] = -1  # Sell
    else:
        df.at[df.index[i], 'Signal'] = 0  # Hold

# Print the DataFrame with signals
print(df)

# Plot the data for visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))
plt.plot(df['Predicted close'], label='Predicted Close', color='blue')
plt.plot(df['Short_MA'], label='Short MA', color='red')
plt.plot(df['Long_MA'], label='Long MA', color='green')
plt.scatter(df.index[df['Signal'] == 1], df['Predicted close'][df['Signal'] == 1], marker='^', color='g', label='Buy Signal', s=100)
plt.scatter(df.index[df['Signal'] == -1], df['Predicted close'][df['Signal'] == -1], marker='v', color='r', label='Sell Signal', s=100)
plt.title('Trading Signals')
plt.xlabel('timestamp')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




