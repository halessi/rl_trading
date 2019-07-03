import gym
import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

import trade_gym
from utils import plot

def main():
    ''' 
    Train an agent. Note that I've created a custom OpenAI Gym environment
    to allow for quick plug and play in comparing performance across 
    different RL models. 

    '''
    env = gym.make('Trade-v0',
                   window = 50, 
                   datadir = 'stocks/aapl_1min.csv',
                   preprocesses = ['MinMax']
                   )

    network_spec = [
        dict(type = 'flatten'),
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ]

    agent = PPOAgent(
        states=env.observation_space,
        actions=env.action_space,
        network=network_spec,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        optimization_steps=10,
        scope='ppo',
        discount=0.99,
        entropy_regularization=0.01,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        likelihood_ratio_clipping=0.2,
    )

    runner = Runner(agent=agent, environment=env)

    def episode_finished(r):
        print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                    reward=r.episode_rewards[-1]))
        return True

    runner.run(episodes=100, episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}. Average reward of last 10 episodes (of 100): {ar}.".format(
            ep=runner.episode,
            ar=np.mean(runner.episode_rewards[-10:]))
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
    
    plot(collectables, 0.1)

if __name__ == '__main__':
    main()
 