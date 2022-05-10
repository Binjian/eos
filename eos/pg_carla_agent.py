""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np

import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt


# hyperparameters
H = 10  # number of hidden layer neurons
batch_size = 1  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False
A = 11  # action_resolution
A2 = A * A  # A square
A3 = A * A * A  # A cubic
# throttle_space = np.linspace(0.0, 1.0, A)
# action space is a table of 3xA, PID parameters with heuristic boundaries
KP_space = np.logspace(-1.0, 1.0, A)
KI_space = np.logspace(-1.0, np.log10(3), A)
KD_space = np.logspace(-2.0, 0.0, A)

# model initialization
D = 2  # input dimensionality: 2 [speed, acc]
H = 4  # number of hidden layer neurons
if resume:
    model = pickle.load(open("data/pg-carla-save.p", "rb"))
else:
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization 4*2
    model["W2"] = np.random.randn(A3, H) / np.sqrt(H)  # 1331*4
    # model["W2P"] = np.random.randn(A, H) / np.sqrt(H)
    # model["W2I"] = np.random.randn(A, H) / np.sqrt(H)
    # model["W2D"] = np.random.randn(A, H) / np.sqrt(H)
# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def discount_rewards(r):
    """take 1D float array of rewards and compute discounted reward"""
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
    print(model["W1"])
    print(x)
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp_pid = np.dot(model["W2"], h)
    # p = sigmoid(logp)
    p_pid = softmax(
        logp_pid
    )  # .reshape(-1, 3)  # three columns corresponding to Kp, Ki, Kd
    return p_pid, h  # return probability of taking action , and hidden state


def policy_backward(epx, eph, epdlogp):  # something wrong here
    """backward pass. (eph is array of intermediate hidden states)"""
    # dW2 = np.dot(eph.T, epdlogp).ravel()
    dW2 = np.dot(epdlogp.T, eph)  # 33*4
    # dh = np.outer(epdlogp, model["W2"])
    dh = np.dot(epdlogp, model["W2"])  # 4*2

    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)  # 4x2
    return {"W1": dW1, "W2": dW2}
