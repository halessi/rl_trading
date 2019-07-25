import argparse
import json

import gym
import numpy as np
from tensorforce.agents import Agent
from tensorforce.execution import Runner

import trade_gym
from utils import plot


def main(args):
    ''' 
    Train an agent. Note that I've created a custom OpenAI Gym environment
    to allow for quick plug and play in comparing performance across 
    different RL models. 

    NOTE: as of now, because states are 2d, I've added an additional flatten layer to the network
    this needs to be fixed if we wanted to do CONVOLUTIONS, which we probably do

    '''
    env = gym.make('Trade-v0',
                   window = args.window, 
                   datadir = 'stocks/s_coinbaseUSD_1_min_data_2014-12-01_to_2018-11-11.csv',
                   preprocesses = [args.preprocess],
    )

    # TODO: probably not wrap this in a try and actually find error in failure
    try:
        with open(args.agent, 'r') as a:
            agent = json.load(fp=a)
    except:
        raise AttributeError('no agent config')
    
    try:
        with open(args.network, 'r') as n:
            network = json.load(fp=n) 
    except:
        raise AttributeError('no network config')

    # prepend the flatten layer, see notes above
    network.insert(0, {'type': 'flatten'})

    agent = Agent.from_spec(spec = agent,
                            kwargs = dict(
                                states = env.observation_space,
                                actions = env.action_space,
                                network = network
                            )
    )

    runner = Runner(agent = agent, environment = env)

    def episode_finished(r):
        print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                    reward=r.episode_rewards[-1]))
        return True

    runner.run(episodes = args.episodes, episode_finished = episode_finished)

    print("Learning finished. Total episodes: {ep}. Average reward of last 10 episodes (of 10): {ar}.".format(
            ep=runner.episode,
            ar=np.mean(runner.episode_rewards[-5:]))
    )

    print('Testing for an episode...')

    s = env.reset()

    collectables = []
    while True:
        action = agent.act(s)
        s, r, d, i = env.step(action)
        agent.observe(reward = r, terminal = d)
        collectables.append((s[0][0], action))          # to be replaced by env.render() when i get it fixed
        if d:
            break
    
    plot(collectables, 0.001) # plot only .1% of one episode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line args...')

    parser.add_argument('-w', '--window', help = 'number of time-steps to use in each state', default = 50)
    parser.add_argument('-p', '--preprocess', help = 'how to preprocess data. options: minMax, ..', default = 'MinMax')
    parser.add_argument('-e', '--episodes', help = 'number of episodes to train for', default = 10)
    parser.add_argument('-a', '--agent', 
        help = 'agent config, examples in configs/agents/ taken from TensorForce at https://github.com/tensorforce/tensorforce/tree/major-revision/examples/configs', 
        default = 'configs/agents/dqn.json')
    parser.add_argument('-n', '--network',
        help = "network architecture (for agent), examples in configs/networks/ taken from same url as above",
        default = 'configs/networks/mlp2_network.json')

    args = parser.parse_args()
    main(args)
