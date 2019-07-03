# rl_trading
I've created this repo out of an effort to better track my explorations of various topics related to algorithmic trading, usually RL-based. **I'll be working over the next couple months** to build this out as a consolidated code-base for all the various side projects I've explored. 

Please reach out if you have any questions!
__hughalessi@gmail.com__

# Installation
If you'd like to make use of this code the current, easiest method for doing so will be cloning the repository and creating a new conda environment based upon the yml file. 

```
git clone https://github.com/halessi/rl_trading
cd rl_trading
conda create -f rl_trading.yml
```

Then activate the new environment with:
```
source activate rl_trading
```

# Training a model
As the code stands now, a TensorForce PPO agent is hardcoded into main.py. To train this agent, run:
```
python main.py
```

# Some results
As can be seen in the images below, ....

Problems with reinforcement learning in time-series challenges struggle from a variety of issues, including delayed credit assignment and a low signal-to-noise ratios. It is unclear whether the OHLCV data we use here, even when preprocessed, contains sufficient signal for advantageously predicting future prices. 

### BTC trading from December 1st, 2014 until ~ March 1st, 2015
![Optional Text](../master/imgs/btc_120114_to_030115.png)

### $AAPL trading from ...
![Optional Text](../master/imgs/AAPL.png)

# Future steps (notes for me!)
- [ ] run with aggregated bitcoin order book data
- [ ] implement here a convolutional network for derivation of more specific state information
- [ ] explore applicability of renko-blocks for denoising time-series data
- [ ] incorporate sentiment analysis into state information
- [ ] set up command-line-args input to enable more rapid model and environment prototyping
- [ ] attempt to use an autoencoder for feature extraction on OHLCV data before feeding to RL
- [X] implement basic MinMax scaling with scikit-learn
- [X] build OpenAI gym environment
