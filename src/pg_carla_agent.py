""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np

# import cPickle as pickle
import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt

# import gym
# from baseline_environm_vanilla_discrete import *

# hyperparameters
H = 10  # number of hidRen layer neurons
batch_size = 4  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False
A = 21  # action_resolution
throttle_space = np.linspace(0.0, 1.0, 21)
# model initialization
D = 2  # input dimensionality: 2 [speed, acc]
H = 4  # number of hidden layer neurons
if resume:
    model = pickle.load(open("pg-carla-save.p", "rb"))
else:
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model["W2"] = np.random.randn(A, H) / np.sqrt(H)
# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(list(range(0, r.size))):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model["W2"], h)
    # p = sigmoid(logp)
    p = softmax(logp)
    return p, h  # return probability of taking action , and hidden state


def policy_backward(epx, eph, epdlogp):  # something wrong here
    """ backward pass. (eph is array of intermediate hidden states) """
    # dW2 = np.dot(eph.T, epdlogp).ravel()
    dW2 = np.dot(epdlogp.T, eph)  # 10x4
    # dh = np.outer(epdlogp, model["W2"])
    dh = np.dot(epdlogp, model["W2"])

    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)  # 4x2
    return {"W1": dW1, "W2": dW2}


# env = Environment()
