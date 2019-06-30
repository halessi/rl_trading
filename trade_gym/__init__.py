from gym.envs.registration import register

register(
    id='Trade-v0',
    entry_point='trade_gym.envs:TradeEnv',
    kwargs={'datadir': 'stocks'}
)
