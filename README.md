**this is in progress. check for periodic updates! **

also: there are so many problems with this, please DO NOT attempt to use it in any live setting.

![Optional Text](../master/imgs/fomo.gif)

# rl_trading
I've created this repo out of an effort to better track my explorations of various topics related to algorithmic trading, usually RL-based. **I'll be working over the next couple months** to build this out as a consolidated code-base for all the various side projects I've explored. 

Please reach out if you have any questions!
__hughalessi@gmail.com__

# Installation
If you'd like to make use of this code the current, easiest method for doing so will be cloning the repository and creating a new conda environment based upon the yml file. 

```
git clone https://github.com/halessi/rl_trading
cd rl_trading
conda env create -f rl_trading.yml
```

Then activate the new environment with:
```
source activate rl_trading
```

# Training a model
TensorForce and OpenAI Gym, two incredible resources for RL-learning, provide a neat and clean 
manner for formalizing the interactions between the agent and environment. This allows the code to configure the agent
as a separate entity from the underlying environment that handles all of the stock-data manipulation. 

For some background, remember that the basic formulation for training a reinforcement learning agent is as follows:
```
# fetch the first state (after initializing env and agent)
initial_state = env.reset()

# act on our first state
action = agent.act(state = initial_state)

# process the action, fetch the next state, reward, and done (which tells us whether the environment is terminal,
# which might occur if the agent ran out of $$)
reward, done, next_state = env.step(action = action)

# act on the next state
action = agent.act(state = next_state)
.....
# continue until finished
```

Using OpenAI's formal structure for an environment, I created stock_gym, which is a work in progress, but eventually will 
allow all manner of underlying preprocessing and data manipulation to enable our agent to extract the greatest signal.

To train an agent, see below. 

```
python main.py /
    --window=      # whatever length of timesteps you'd like each state to include
    --preprocess=  # how you'd like the data to be preprocessed, options (TO EVENTUALLY) include: [MinMax, Renko log-return, autoencoded, hopefully more]
    --episodes=    # the number of episodes to train for. an episode is a full run through the dataset
    --agent=       # agent, how actions are chosen from network results
    --network=     # network architecture for data analysis
```

# Some results
As can be seen in the images below, ....

Problems with reinforcement learning in time-series challenges struggle from a variety of issues, including delayed credit assignment and a low signal-to-noise ratios. It is unclear whether the OHLCV data we use here, even when preprocessed, contains sufficient signal for advantageously predicting future prices. (data MinMax processed, hasn't been unscaled here. immediate next step is to implement log-return scaling)

### $AAPL trading example
![Optional Text](../master/imgs/AAPL.png)

# Future steps (notes for me!)
- [ ] run with aggregated bitcoin order book data
- [ ] implement here a convolutional network for derivation of more specific state information
- [ ] explore applicability of renko-blocks for denoising time-series data
- [ ] incorporate sentiment analysis into state information
- [ ] attempt to use an autoencoder for feature extraction on OHLCV data before feeding to RL
- [ ] implement log-return scaling
- [X] set up command-line-args input to enable more rapid model and environment prototyping
- [X] implement basic MinMax scaling with scikit-learn
- [X] build OpenAI gym environment
