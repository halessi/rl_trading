import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gym import spaces


class TradeEnv(gym.Env):
    '''Trading environment. 

    Arguments:
        window: int ~ 50/100. Data points to include per observation. 
        datasource: string, either 'local', 'robinhood', or 'iex'. Source datapoints. 
        preprocess: string, either 'None', 'renko', or 'log_transform'. Preprocess datapoints. 
        datadir: if using datasource of 'local', need to provide datadir.
        reward: whether to use portfolio ROI or sharpe ratio as reward.

    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, window = 50, datasource = 'local', preprocesses = ['None'], datadir = None, reward = 'ROI',
                 use_market_profile = False):
        self.window = window
        self.reward_meth = reward
        self.action_space = spaces.Discrete(3)
        self.use_market_profile = use_market_profile
        self.preprocesses = preprocesses
        self.fig = None

        '''FIXME: is this correct? '''
        self.commission = 0.1 / 100

        if datasource == 'local':
            if datadir != datadir: # check for Nonetype
                raise ValueError('Error: please specify data directory.')
            else:
                self.data = self.load_normal(datadir = datadir)
        elif datasource == 'robinhood':
            return NotImplementedError
        elif datasource == 'iex':
            return NotImplementedError

        for preprocess in preprocesses:
            if preprocess == 'None':
                pass
            elif preprocess == 'MinMax':
                self.data = self.preprocess_MinMax()
            elif preprocess == 'renko':
                self.data = self.preprocess_renko()
            elif preprocess == 'log_transform':
                self.data = self.preprocess_log_transform()
            elif preprocess == 'autoencode':
                self.data = self.preprocess_autoencode()
        
        if self.observation_space is None:
            self.observation_space = spaces.Box(low = 0, high = 10000, shape = (self.window, self.data.shape[1]))
            print(self.data.head(n=5))

    def reset(self):
        self.cash = 1e6 # 1,000,000 to start
        self.equity = 0
        self.steps = self.window
        self.equity = 0

        return self.get_next_state()
    
    def preprocess_MinMax(self):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range = (0,1))
        self.columns = self.data.columns
        self.data = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(self.data, columns = self.columns)
        return pd.DataFrame(self.data)

    def preprocess_renko(self):
        return NotImplementedError

    def preprocess_log_transform(self):
        return NotImplementedError

    def preprocess_autoencode(self):
        from keras.models import load_model

        # set the observation space size to output of the encoder
        model = load_model('encoder/conv_encoder.h5')
        self.ae = model
        output_shape = model.layers[-1].output_shape

        '''XXX: might not be necessary? '''
        if len(output_shape) > 1:
            self.observation_space = spaces.Box(low = 0, high = 10000, shape = output_shape[1:])
        elif len(output_shape) == 1:
            self.observation_space = spaces.Box(low = 0, high = 10000, shape = (output_shape[1],))

        return self.data

    def load_normal(self, datadir):
        dtypes = {'Date': str, 'Time': str}
        df = pd.read_csv(datadir, sep=',', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype=dtypes, engine = 'python')
        df.index = pd.to_datetime(df.Time)

        df = df.iloc[0:1000]

        if self.use_market_profile: 
            self.mp = MarketProfile(df, mode = 'tpo')

            for i in range(0, df.shape[0] - self.window - 1):
                try:
                    mp_slice = self.mp[i:i+self.window].as_dict()
                except KeyError:
                    # XXX: not sure if behavior here is ideal
                    print('error: {}'.format(df['Time'][i+self.window]))
                for key, value in mp_slice.items():
                    df.ix[i+self.window, key] = value

        try: 
            df = df.iloc[self.window:]
            df.fillna(method = 'ffill', inplace = True)
            df.drop(['Date', 'Time'], axis = 1, inplace = True)
        except AttributeError:
            pass

        print('Data loaded, shape {}.'.format(df.shape))
        print('Number of null values after dropping: {}.'.format(df.isnull().sum().sum()))
        return df

    def get_current_price(self):
        return self.data['Close'][self.steps]

    def get_next_state(self):
        state = self.data[self.steps:self.steps + self.window]
        self.image = state           

        # try to reshape state array to fit match "Input" layer of autoencoder
        if 'autoencode' in self.preprocesses:
            shape = self.ae.layers[0].input.shape[1:]
            if len(shape) > 1:
                state = np.array(state).reshape(shape)
            elif len(shape) == 1:
                state = np.array(state).flatten()
            state = self.ae.predict(np.expand_dims(state, axis = 0))[0]

        assert state.shape == self.observation_space.shape, print('Error, observation shape incorrect: {}. Should be: {}'.format(state.shape, 
                                                                                                                self.observation_space.shape))                                                                                                                                                                                                                                                                  
        try:
            return state.values
        except AttributeError: # already numpy array
            return state

    def step(self, action):
        assert self.action_space.contains(action)

        curr_price = self.get_current_price()

        # determine portfolio value
        portfolio = self.cash + \
                    (1 - self.commission) * self.equity \
                    * curr_price

        '''NOTE: 0 = hold, 1 = buy, 2 = sell'''
        if action == 1:
            if self.cash >= (curr_price * (1. + self.commission)): 
                self.equity += 1 # add one share
                self.cash -= (curr_price * (1. + self.commission))
            else:
                done = True # kill early

        elif action == 2:
            if self.equity > 0:
                self.cash += (curr_price * (1. - self.commission))
                self.equity -= 1 # subtract one share

        '''iterate to next state'''
        self.steps += 1
        new_portfolio = self.cash + \
                        (1 - self.commission) * self.equity \
                        * self.get_current_price()

        # terminate if out of states NOTE: this will need to be changed a lot for live
        done = True if self.data.shape[0] < self.steps + self.window + 1 else False
        
        if self.reward_meth == 'ROI':
            reward = new_portfolio - portfolio
        elif self.reward_meth == 'Sharpe':
            # TODO: implement
            pass

        info = {
            'portfolio' : new_portfolio,
            'cash': self.cash,
            'assets': self.equity,
        }

        return self.get_next_state(), reward, done, info

    def render(self, mode):
        if mode == 'human':
            _img = pd.DataFrame(self.scaler.inverse_transform(self.image), columns = self.columns)

            if self.fig is None:
                self.fig, self.axes = plt.subplots(nrows = len(self.columns), ncols = 1)
                for counter, col in enumerate(self.columns):
                    self.axes[counter].plot(_img[col], label = col)
                plt.show(block = True)
                plt.pause(1e-8)
            else:
                for counter, col in enumerate(self.columns):
                    self.axes[counter].set_data(range(len(_img[col])), _img[col])
                plt.pause(1e-8)
            

if __name__ == '__main__':
    env = TradeEnv(window = 50, 
                   datadir = 'stocks/aapl_1min.csv', 
                   preprocesses = ['MinMax'])
    env.reset()
    done = False
    while done == False:
        obs, r, done, i = env.step(0)
        env.render(mode = 'human')

    # env.reset()
    # done = False
    # obs = []
    # while done == False:
    #     observation, reward, done, info = env.step(0)
    #     obs.append(observation)
    # obs = np.array(obs)
    # print(obs.shape)
    # np.save('train.npy', obs)
