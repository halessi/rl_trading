import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot(collectables, portion):
    '''
    quick bandaid for broken render feature
    given a list of (price, action) in the form collectables, create an 
    image of what the agent did over time
    '''
    # plot over just a specified span
    collectables = collectables[0: round(len(collectables) * portion)]

    # sort actions into buys and sells for scatterplot
    buys, sells = [], []
    for timepoint in range(len(collectables)):
        if collectables[timepoint][1] == 1:    buys.append((timepoint, collectables[timepoint][0]))
        elif collectables[timepoint][1] == 2:  sells.append((timepoint, collectables[timepoint][0]))
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    l, = ax.plot(range(0, len(collectables)), [i[0] for i in collectables])
    buys  = ax.scatter(*zip(*buys), c = 'green', marker = '^')
    sells = ax.scatter(*zip(*sells), c =  'red', marker = 'v')
    ax.set_title('Price')
    ax.grid('on')

    # save plot
    plt.savefig('imgs/BTC')
    plt.show()

def print_data_info(data):
    '''
    pretty print method for training data/info, 
    easy way to confirm things loaded correctly
    '''

    print('*'*60)
    print('\n')
    print('Data loaded, shape {}.'.format(data.shape) + ' Data columns: {}.'.format(data.columns))
    print('Number of null values after dropping: {}. (If this is >0, reward will be NaN.)'.format(data.isnull().sum().sum()))
    print('\n')
    print(data.head(n=5))
    print('\n')
    print('*'*60)
    
