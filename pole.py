#!/usr/bin/env python3

import sys
import numpy as np
import gym
from sklearn.neighbors import KDTree

params = {'training_episodes': 1000, 'n_neighbors': 10, 'testing_episodes': 1000, 'discount': 1}
for arg in sys.argv[1:]:
  a, b = arg.split('=')
  if '.' in b:
    params[a] = float(b)
  else:
    params[a] = int(b)

env = gym.make('CartPole-v0')

def gen_data(n_episodes = 1):
  d = []
  for _ in range(n_episodes):
    c = []
    obs = env.reset()
    done = False
    r = 0
    discount = 1
    while not done:
      random_action = 0 if np.random.random() < 0.5 else 1
      new_obs, reward, done, info = env.step(random_action)
      c += [np.array(obs.tolist() + [random_action, reward])]
      obs = new_obs
      r += reward * discount
      discount *= params['discount']
    for x in c:
      x[-1] = r
    d += c
  return np.array(d)

def test(n_episodes, foo):
  r = []
  for _ in range(n_episodes):
    obs = env.reset()
    done = False
    reward = 0
    while not done:
      action = foo(obs)
      obs, cur_reward, done, info = env.step(action)
      reward += cur_reward
    r += [reward]
  return np.mean(r)

data = gen_data(params['training_episodes'])
actions = data[:,-2]
rewards = data[:,-1]
observations = data[:,:-2]
del data

kd_tree = KDTree(observations)
def kd_action(obs):
  _, ind = kd_tree.query(obs.reshape((1, -1)), k = params['n_neighbors'])
  r = [0, 0]
  c = [1e-9, 1e-9]
  for i in ind[0,]:
    r[int(actions[i])] += rewards[i]
    c[int(actions[i])] += 1
  return np.argmax([r[0] / c[0], r[1] / c[1]])

r = test(params['testing_episodes'], kd_action)
if r >= 195:
  print(r, '*')
else:
  print(r)

