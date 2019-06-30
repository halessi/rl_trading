## rl_trading
A playground for all things reinforcement learning in trading!

# Installation
If you'd like to make use of this code, the easiest method for doing so will be cloning the repository and creating a new conda environment based upon the yml file. 

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
As the code stands now, a TensorForce PPO agent is hardcoded into main.py. To train this agent, run
```
python main.py
```

# Some results
As can be seen in the images below, ....

Problems with reinforcement learning in time-series challenges like this involved delayed credit-assignment and a fundamental lack of signal. It is unclear whether the OHLCV data we use here, even when preprocessed, contains any signal.  

# Future steps
- [ ] set up command-line-args inputs to enable more rapid model and environment prototyping
- [ ] attempt to use an autoencoder for feature extraction on OHLCV data before feeding to RL
- [X] implement basic MinMax scaling with scikit-learn
- [X] build OpenAI gym environment
