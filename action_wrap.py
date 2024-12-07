import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import queue
from time import sleep

class ComboWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ComboWrapper, self).__init__(env)
        self.combo_list = [
            # [[0,1], [0,4], [0,0]], # Hiza-Geri
            # [[5,7], [0,0]], # Seoi-Nage
            # [[1,7], [0,0]], # Jigoku-Guruma
            # [[1,5], [0,0]], # Inazuma-Kakato-Wari
            # [[0,2], [0,3], [0,0]], # Target-Combo
            # special Moves
            [[7,0], [6,0], [5,0], [0,3], [0,0]], # Hadoken
            [[7,0], [8,0], [1,0], [0,3], [0,0]], # Hadoken
            [[7,0], [6,0], [5,0], [0,6], [0,0]], # Hadoken
            [[7,0], [8,0], [1,0], [0,6], [0,0]], # Hadoken
            # super art
            [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,3], [0,0]], # Shoryu-Reppa
            [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,3], [0,0]], # Shoryu-Reppa
            [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,0]], # Shinryu-Ken
            [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,0]], # Shinryu-Ken
            [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
            [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
        ]

    def step(self, action):
        if action[0] == 9:
            # push combo in
            combo = queue.Queue()
            combo_idx = action[1]
            combo_to_use = self.combo_list[combo_idx]
            for i in range(len(combo_to_use)):
                combo.put(combo_to_use[i])
            combo_reward = 0.0
            done = False
            info = None
            obs = None
            truncated = False
            # reward_record = []
            recv = False
            while(not combo.empty()):
                obs, reward, done, truncated, info = self.env.step(combo.get())
                # reward_record.append(reward)
                combo_reward += reward
                if reward > 0: recv = True
                if done or truncated or reward < 0 or (recv and reward == 0):
                    break
            # if combo_reward != 0:
            #     print('combo', combo_idx, ': ', reward_record)
            #     sleep(0.5)
            return obs, combo_reward, done, truncated, info
        else:
            obs, reward, done, truncated, info = self.env.step(action)
            # if reward != 0:
            #     print('ordinary: ', reward)
            #     sleep(0.5)
            return obs, reward, done, truncated, info

class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MultiDiscreteToDiscreteWrapper, self).__init__(env)

        # Ensure the original action space is MultiDiscrete
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        # self.combo_list = [
        #     # [[0,1], [0,4], [0,0]], # Hiza-Geri
        #     # [[5,7], [0,0]], # Seoi-Nage
        #     # [[1,7], [0,0]], # Jigoku-Guruma
        #     # [[1,5], [0,0]], # Inazuma-Kakato-Wari
        #     # [[0,2], [0,3], [0,0]], # Target-Combo
        #     # special Moves
        #     [[7,0], [6,0], [5,0], [0,3], [0,0]], # Hadoken
        #     [[7,0], [8,0], [1,0], [0,3], [0,0]], # Hadoken
        #     [[7,0], [6,0], [5,0], [0,6], [0,0]], # Hadoken
        #     [[7,0], [8,0], [1,0], [0,6], [0,0]], # Hadoken
        #     # super art
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,3], [0,0]], # Shoryu-Reppa
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,3], [0,0]], # Shoryu-Reppa
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,0]], # Shinryu-Ken
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,0]], # Shinryu-Ken
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
        # ]

        # # Store the original MultiDiscrete action space
        # self.multi_discrete_space = env.action_space

        # # Calculate the total number of discrete actions needed
        # self.n_discrete_actions = np.prod(self.multi_discrete_space.nvec)

        # # Define the new discrete action space
        # self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.action_space = spaces.MultiDiscrete([10, 10])
        # self.reset_without_seed = copy.deepcopy(env.reset)
        # self.combo = queue.Queue()

    def action(self, action):
        return action
        # # print(action.shape)
        # if self.combo.empty():
        #     if action[0] < 9:
        #         return action
        #     else:
        #         # push combo in
        #         combo_idx = action[1]
        #         combo_to_use = self.combo_list[combo_idx]
        #         for i in range(len(combo_to_use)):
        #             self.combo.put(combo_to_use[i])
        #         return self.combo.get()
        # else:
        #     return self.combo.get()

    
    # def reset(self, seed):
    #     self.seed(seed)
    #     self.reset_without_seed()
    #     return self.state

    # def reverse_action(self, action):
    #     # Convert multi-discrete action to discrete action
    #     discrete_action = 0
    #     multiplier = 1
    #     for act, n in zip(reversed(action), reversed(self.multi_discrete_space.nvec)):
    #         discrete_action += act * multiplier
    #         multiplier *= n
    #     return discrete_action

class CustomMultiDiscreteEnv(gym.Env):
    def __init__(self, env):
        super(CustomMultiDiscreteEnv, self).__init__()
        # Your initialization code here

    def reset(self, seed):
        self.seed(seed)
        self.reset()
        return self.state



# # Example usage with a custom environment
# class CustomMultiDiscreteEnv(gym.Env):
#     def __init__(self):
#         super(CustomMultiDiscreteEnv, self).__init__()
#         self.action_space = spaces.MultiDiscrete([2, 3, 2])  # Example multi-discrete action space
#         self.observation_space = spaces.Discrete(5)  # Example observation space
#
#     def step(self, action):
#         # Implement the environment's step function
#         obs = self.observation_space.sample()
#         reward = 1.0
#         done = False
#         info = {}
#         return obs, reward, done, info
#
#     def reset(self):
#         # Implement the environment's reset function
#         return self.observation_space.sample()
#
#
# # Instantiate the custom environment
# env = CustomMultiDiscreteEnv()
#
# # Wrap the environment to convert MultiDiscrete actions to Discrete actions
# wrapped_env = MultiDiscreteToDiscreteWrapper(env)
#
# # Test the wrapped environment
# obs = wrapped_env.reset()
# done = False
#
# while not done:
#     action = wrapped_env.action_space.sample()
#     obs, reward, done, info = wrapped_env.step(action)
#     wrapped_env.render()
#
# wrapped_env.close()
