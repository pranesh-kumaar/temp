#!/usr/bin/env python
# coding: utf-8

# In[500]:


import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential


# In[533]:


#loading the signals
import pandas as pd
signals = pd.read_excel(r'data_with_signals/AAPL.xlsx')
# signals = signals['Unnamed: 0'].rename('Date')
signals = signals.drop(['Trade'], axis = 1)
signals = signals.set_index('Date')


# In[749]:


from ta.volume import VolumeWeightedAveragePrice, chaikin_money_flow, money_flow_index, volume_price_trend
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.volatility import average_true_range
from ta.trend import adx, ema_indicator, macd, macd_diff, macd_signal, adx_neg, adx_pos
from ta.others import daily_log_return

df = signals
df['vwap'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window =14).volume_weighted_average_price()
df['rsi'] = RSIIndicator(close=df["Close"], window = 14).rsi()/100
# df['rsi_log'] = daily_log_return(df['rsi'])
df['stochrsi'] = StochRSIIndicator(close=df["Close"], window = 14).stochrsi()
df['stochrsi_k'] = StochRSIIndicator(close=df["Close"], window = 14).stochrsi_k()
df['stochrsi_d'] = StochRSIIndicator(close=df["Close"], window = 14).stochrsi_d()
df['stoch'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window =14, smooth_window = 3).stoch()/100
df['stoch_signal'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window =14, smooth_window = 3).stoch_signal()/100
df['chaikin_money_flow'] = chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window =14)
df['money_flow_index'] = money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window =14)/100
# df['volume_price_trend'] = volume_price_trend(df['Close'], df['Volume'])
df['average_true_range'] = average_true_range(df['High'], df['Low'], df['Close'], window =14)/12
df['adx'] = adx(df['High'], df['Low'], df['Close'], window =14)/100
df['adx_neg'] = adx_neg(df['High'], df['Low'], df['Close'], window =14)/100
df['adx_pos'] = adx_pos(df['High'], df['Low'], df['Close'], window =14)/100
df['ema'] = ema_indicator(df['Close'], window =14)
df['close-ema'] = df['Close']-df['ema']
df['close-ema-normalized']  = 2*(df['close-ema'] - (df['close-ema'].min()))/(df['close-ema'].max() - (df['close-ema'].min())) - 1
df['macd'] = (macd(df['Close'], 12, 26, 9) - (-10))/(40 - (-10))
df['macd-copy'] = 2*(df['macd'] - (df['macd'].min()))/(df['macd'].max() - (df['macd'].min())) - 1
df['macd_diff'] = (macd_diff(df['Close'], 12, 26, 9) - (-10))/(40 - (-10))
df['macd_signal'] = (macd_signal(df['Close'], 12, 26, 9) - (-10))/(40 - (-10))
df['macd_gap'] = df['macd'] - df['macd_signal']
df['pct_open'] = (df['Open']).pct_change()
df['pct_high'] = (df['High']).pct_change()
df['pct_low'] = (df['Low']).pct_change()
df['pct_close'] = (df['Close']).pct_change()
df['pct_vwap'] = (df['vwap']).pct_change()

df = df.drop(['Open', 'High', 'Low', 'Close', 'vwap', 'macd', 'Volume', 'Signal', 'ema', 'close-ema'], axis = 1)
df = df.iloc[35:]
df = df.tail(75) #For testing purposes
df.to_excel(r'Parameters.xlsx')


# In[772]:


# Neuron Layer 1 creation:
look_back = 14
neuron_cycle = []
for x in range(0, len(df.index)-look_back):
    n_df = df.iloc[x:look_back+x]
    n1 = []
    for y in n_df:
        for z in n_df[y]:
            n1.append(z)
            
    neuron_cycle.append(n1)


# In[773]:


#Generating random Weights
import random

def random_weights():
    
    global no_of_neurons, weights
    no_of_neurons = [32,32,1] #number of random weights for layer 2, layer 3, layer 4
    w1 = []
    w2 = []
    w3 = []
    for x in no_of_neurons:
        for i in range(0,x):
            w1.append(random.uniform(-1, 1)/100)
            w2.append(random.uniform(-1, 1))
            w3.append(random.uniform(-1, 1)/10)
    weights = np.array([w1,w2,w3])
    
    layer_weights[individual] = weights


# In[774]:


# Neuron Layer 2 creation through weights using genetic algorithm:

def layer_2():
    
    global n2
    n2 = []
    for i in range(0,no_of_neurons[0]):
        y1 = []
        for x in n1:
            w = layer_weights[individual][0][i]
            z = w*x
            y1.append(z)
        n2.append(sum(y1))

    n2 = np.tanh(np.array(n2))


# In[775]:


# Neuron Layer 3 creation through weights using genetic algorithm:

def layer_3():
    
    global n3
    n3 = []
    for i in range(0,no_of_neurons[1]):
        y2 = []
        for x in n2:
            w = layer_weights[individual][1][i]
            z = w*x
            y2.append(z)
        n3.append(sum(y2))

    n3 = np.tanh(np.array(n3))


# In[776]:


#Final Neuron Layer - Output
    
def layer_4():
    
    global n4
    n4 = []
    for i in range(0,no_of_neurons[2]):
        y3 = []
        for x in n3:
            w = layer_weights[individual][2][i]
            z = w*x*10
            y3.append(z)
        n4.append(sum(y3))

    n4 = np.tanh(np.array(n4))
    
    if n4 >= 0.5 :
        n4 = 1
    elif n4 <= -0.5 :
        n4 = -1
    else: 
        n4 = 0
    
    return n4


# In[777]:


#Creating First Generation of Trade Signals

population = 10
signal_population = []
layer_weights = [0]*population

global individual

for individual in range(0,population):
    signal =[]
    for z in neuron_cycle: 
        n1 = z
        random_weights()
        layer_2()
        layer_3()
        output = layer_4()
        signal.append(output)
        
        # layer_weights[x] = weights
        
    signal_population.append(signal)
    
#layer_weights[y][x] contains y layer weights of generation x


# In[778]:


#Setting rules for backtesting

from backtesting import Backtest, Strategy

class SignalStrategy(Strategy):
    def init(self):
        pass
    
    def next(self):
        global index
        index = []
        current_signal = self.data.Signal[-1]
        if current_signal == 1:
            if not self.position:
                self.buy()
        elif current_signal == -1:
            if self.position:
                self.position.close()


# In[779]:


import threading
import warnings
warnings.filterwarnings('ignore')

backtest_data = signals.tail(len(signal))

results = []
for x in signal_population:
    backtest_data['Signal'] = x
    
    bt = Backtest(backtest_data, SignalStrategy, cash = 10_000)
    stats = bt.run()

    # bt.plot()
    # print (stats) #stats[6] gives the returns
    
    results.append(stats[6])
    
sort = np.argsort(results)[::-1][:population]

sorted_results = []
sorted_weights = []
for x in sort:
    sorted_results.append(results[x])
results


# In[758]:


#Mutation

mutation_rate = 0.2
drop_rate = 0.4

drop = []
i = 1
for x in reversed(sort):
    drop.append(x)
    if i >= (len(sort)*drop_rate) :
        break
    i = i+1
    
#layer_weights[x][y] contains y layer weights of generation x

#dropping bottom 4 generation
# for x in sorted(drop, reverse = True):
#     del layer_weights[x]


# for x in range(0,len(layer_weights)):
#     for y in range(0,len(layer_weights[x])):
#         for z in range(0,len(layer_weights[x][y])):
#             var = layer_weights[x][y][z]
#             layer_weights[x][y][z] = random.uniform(var*(1-mutation_rate), var*(1+mutation_rate))


# In[781]:


#Creating Second generation after mutating

signal_population = []
for x in range(0,len(layer_weights)):
    signal =[]
    weights = [layer_weights[x][0],layer_weights[x][1],layer_weights[x][2]]
    for z in neuron_cycle: 
        n1 = z
        # random_weights()
        # print (weights)
        layer_2()
        layer_3()
        output = layer_4()
        signal.append(output)

    signal_population.append(signal)
    
#layer_weights[y][x] contains y layer weights of generation x
np.array(signal_population)


# In[765]:


import threading
import warnings
warnings.filterwarnings('ignore')

backtest_data = signals.tail(len(signal))

results = []
for x in signal_population:
    backtest_data['Signal'] = x
    
    bt = Backtest(backtest_data, SignalStrategy, cash = 10_000)
    stats = bt.run()

    # bt.plot()
    # print (stats) #stats[6] gives the returns
    
    results.append(stats[6])
    
sort = np.argsort(results)[::-1][:population]

sorted_results = []
sorted_weights = []
for x in sort:
    sorted_results.append(results[x])
    
print (results)


# In[ ]:




