""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np

# import cPickle as pickle
import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt

# import gym
from baseline_environm_vanilla_discrete import *

# hyperparameters
H = 10  # number of hidRen layer neurons
batch_size = 4  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False
A = 3  # action_resolution
# model initialization
D = 2  # input dimensionality: 2
H = 4  # number of hidden layer neurons
if resume:
    model = pickle.load(open("veos-save.p", "rb"))
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


def policy_backward(eph, epdlogp):  # something wrong here
    """ backward pass. (eph is array of intermediate hidden states) """
    # dW2 = np.dot(eph.T, epdlogp).ravel()
    dW2 = np.dot(epdlogp.T, eph)  # 10x4
    # dh = np.outer(epdlogp, model["W2"])
    dh = np.dot(epdlogp, model["W2"])

    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)  # 4x2
    return {"W1": dW1, "W2": dW2}


env = Environment()

[observation, reward, done, info] = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:

    # preprocess the observation, set input to network to be difference image
    x = observation

    # forward the policy network and sample an action from the returned
    # probability
    aprob, h = policy_forward(x)
    action = np.random.choice(A, 1, aprob.tolist())[0]

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    # y = 1 if action == 2 else 0 # a "fake label"
    y = np.zeros(A)
    y[action] = 1
    # grad that encourages the action that was taken to be taken
    dlogps.append(y - aprob)
    # (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    if observation[0] > 99:
        print(observation[0])
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for
    # previous action)
    drs.append(reward)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and
        # rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient
        # estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in list(model.items()):
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = (
                    decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                )
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                # reset batch gradient buffer
                grad_buffer[k] = np.zeros_like(v)

        # boring book-keeping
        running_reward = (
            reward_sum
            if running_reward is None
            else running_reward * 0.99 + reward_sum * 0.01
        )
        print(
            "resetting env. episode %d reward total was %f. running mean: %f"
            % (episode_number, reward_sum, running_reward)
        )
        if episode_number % 100 == 0:
            pickle.dump(model, open("save.p", "wb"))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
