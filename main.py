import gym
import trade_gym
import tensorforce

def main():
    ''' 
    Train an agent. Note that I've created a custom OpenAI Gym environment
    to allow for quick plug and play in comparing performance across 
    different RL models. 

    '''
    env = gym.make('Trade-v0')


if __name__ == '__main__':
    main()
    