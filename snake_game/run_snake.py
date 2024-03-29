import gymnasium as gym
from train_snake import create_neural_network


env = gym.make('snake-v1', window_size=(500, 400))
dqn = create_neural_network(env)
dqn.load_weights('dqn_snake_weights-bkp.h5f')
dqn.test(env, nb_episodes=1, visualize=True)
