"""
Title: Advantage Actor Critic Method
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2021/02/12
Last modified: 2020/03/15
Description: Implement Advantage Actor Critic Method in Carla environment.
"""
"""
## Introduction

This script shows an implementation of Advantage Actor Critic method on gym-carla environment.

### Actor Critic Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:

1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.

Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.

### Gym-Carla env 

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption 

### References

- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)

"""
"""
## Setup
"""

# drl import
import datetime
import gym
import gym_carla
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

tfd = tfp.distributions

# communcation import
import rospy
import std_msgs.msg
from comm.vcu.msg import *
from threading import Lock
import _thread as thread, queue, time

# visualization import
import pandas as pd
import matplotlib.pyplot as plt
from visualization.visual import plot_to_image

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import io  # needed by convert figure to png in memory

# internal import
from comm.vcu_calib_generator import (
    generate_vcu_calibration,
    generate_lookup_table,
)


# from communication import carla_ros
from comm.carla_ros import talker
from agent.ac_gaussian import (
    constructactorcriticnetwork,
    train_step,
)

from comm.tbox.scripts.tbox_sim import *

set_tbox_sim_path("/home/is/devel/newrizon/drl-carla-manual/src/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# TODO add vehicle communication insterface
# TODO add model checkpoint episodically unique


# multithreading initialization
data_lock = Lock()
vcu_output = VCU_Output()
vcu_input = VCU_Input()
# flash_mutex = thread.allocate_lock()

# TODO add visualization and logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# dataQueue contains a table which is a list of type float
dataQueue = queue.Queue()
# figQueue is for visualization thread to show the calibration 3d figure
figQueue = queue.Queue()
episode_done = False
episode_count = 0


# this is the figure consumer for visualization
def show_calib_table(figqueue):
    while True:
        time.sleep(0.1)
        try:
            figure = figqueue.get()  # default block = True
        except queue.Empty:
            pass
        else:
            plt.show()
            time.sleep(1)
            if episode_done:
                with train_summary_writer.as_default():
                    tf.summary.image(
                        "Calibration Table", plot_to_image(figure), episode_count
                    )
            else:
                plt.close(figure)
            # print("show the calib table!")
            # send_float_array('TQD_trqTrqSetECO_MAP_v', table)


# flash_mutex.acquire()
# this is the calibration table consumer for flashing
def flash_vcu(dataqueue):
    while True:
        time.sleep(0.1)
        try:
            table = dataqueue.get()  # default block = True
        except queue.Empty:
            pass
        else:
            print("flash vcu calib table!")
            # send_float_array('TQD_trqTrqSetECO_MAP_v', table)


def get_torque(data):
    # rospy.loginfo(rospy.get_caller_id() + "vcu.rc:%d,vcu.torque:%f", data.rc, data.tqu)
    with data_lock:
        vcu_output.header = data.header
        vcu_output.torque = data.torque


# TODO add a thread for send_float_array
# TODO add printing calibration table
# TODO add initialize table to EP input
def main():
    global episode_done, episode_count
    # ros msgs for vcu communication
    rospy.init_node("carla", anonymous=True)
    rospy.Subscriber("/newrizon/vcu_output", VCU_Output, get_torque)
    pub = rospy.Publisher("/newrizon/vcu_input", VCU_Input, queue_size=10)

    # parameters for the gym_carla environment
    params = {
        "number_of_vehicles": 0,
        "number_of_walkers": 0,
        "display_size": 256,  # screen size of bird-eye render
        "max_past_step": 1,  # the number of past steps to draw
        "dt": 0.1,  # time interval between two frames
        "discrete": False,  # whether to use discrete control space
        "discrete_acc": [-3.0, 0.0, 3.0],  # discrete value of accelerations
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
        "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
        "ego_vehicle_filter": "vehicle.carlamotors.carlacola",  # filter for defining ego vehicle
        "carlaserver": "192.168.8.100",  # carla server ip address
        "port": 2000,  # connection port
        "town": "town01",  # which town to simulate
        "task_mode": "random",  # mode of the task, [random, roundabout (only for town03)]
        "max_time_episode": 1000,  # maximum timesteps per episode
        "max_waypt": 12,  # maximum number of waypoints
        "obs_range": 32,  # observation range (meter)
        "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
        "d_behind": 12,  # distance behind the ego vehicle (meter)
        "out_lane_thres": 2.0,  # threshold for out of lane
        "desired_speed": 8,  # desired speed (m/s)
        "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
        "display_route": True,  # whether to render the desired route
        "pixor_size": 64,  # size of the pixor labels
        "pixor": False,  # whether to output pixor observation
        "file_path": "../data/highring.xodr",
        "map_road_number": 1,
        "AI_mode": True,
    }

    # configuration parameters for the whole setup
    seed = 42
    gamma = 0.99  # discount factor for past rewards
    max_steps_per_episode = 10000  # maximal steps per episode, 120s / 0.05 is 2400. todo adjust numbers of maximal steps

    env = gym.make("carla-v0", params=params)
    obs = env.reset()

    env.seed(seed)
    eps = np.finfo(np.float64).eps.item()  # smallest number such that 1.0 + eps != 1.0

    """
    ## implement actor critic network
    
    this network learns two functions:
    
    1. actor: this takes as input the state of our environment and returns a
    probability value for each action in its action space.
    2. critic: this takes as input the state of our environment and returns
    an estimate of total rewards in the future.
    
    in our implementation, they share the initial layer.
    """
    vcu_calib_table_col = 17  # number of pedal steps, x direction
    vcu_calib_table_row = 21  # numnber of velocity steps, y direction
    vcu_calib_table_budget = (
        0.05  # interval that allows modifying the calibration table
    )
    vcu_calib_table_size = vcu_calib_table_row * vcu_calib_table_col
    pedal_range = [0, 1.0]
    velocity_range = [0, 20.0]

    # default table
    vcu_calib_table0 = generate_vcu_calibration(
        vcu_calib_table_col, pedal_range, vcu_calib_table_row, velocity_range, True
    )
    vcu_calib_table = np.copy(vcu_calib_table0)  # shallow copy of the default table
    # vcu_act_list = vcu_calib_table.numpy().reshape(-1).tolist()

    # Start thread for flashing vcu, flash first
    thread.start_new_thread(flash_vcu, (dataQueue,))
    thread.start_new_thread(show_calib_table, (figQueue,))
    # flash_mutex.acquire()

    vcu_lookup_table = generate_lookup_table(
        pedal_range, velocity_range, vcu_calib_table
    )

    num_observations = 3  # observed are the current speed and acceleration, throttle
    sequence_len = 20  # 20 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1 second
    num_inputs = num_observations * sequence_len  # 60 subsequent observations
    num_actions = vcu_calib_table_size  # 17*21 = 357
    num_hidden = 128

    # todo connect gym-carla env, collect 20 steps of data for 1 second and update vcu calib table.

    # create actor-critic network
    bias_mu = 0.0  # bias 0.0 yields mu=0.0 with linear activation function
    bias_sigma = 0.55  # bias 0.55 yields sigma=1.0 with softplus activation function
    # checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
    # checkpoint_path = "../tf_ckpts/checkpoint"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    tf.keras.backend.set_floatx("float64")
    # TODO option fix sigma, just optimize mu
    # TODO create testing scenes for gathering data
    actorcritic_network = constructactorcriticnetwork(
        num_observations, sequence_len, num_actions, num_hidden, bias_mu, bias_sigma
    )
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # add checkpoints manager
    checkpoint_dir = "../tf_ckpts"
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), optimizer=opt, net=actorcritic_network
    )
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch")
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # if latest != None:
    #     actorcritic_network = keras.models.load_model(latest)
    """
    ## train
    """

    vcu_action_history = []
    mu_sigma_history = []
    vcu_critic_value_history = []
    vcu_rewards_history = []
    vcu_states_history = []
    running_reward = 0
    episode_reward = 0

    wait_for_reset = True
    L045B = True
    while True:  # run until solved
        # logictech g29 default throttle 0.5,
        # after treading pedal of throttle and brake,
        # both will be reset to zero.
        if wait_for_reset:
            # obs = env.get_init_state()
            obs = env.reset()
            if np.fabs(obs[2] - 0.5) < eps:
                continue
            else:
                wait_for_reset = False
        vcu_reward = 0
        vcu_states = []
        vcu_states.append(obs)
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                if L045B:
                    # while vcu_input.stamp > vcu_output.stamp:
                    talker(
                        pub, env.counter, obs[0], obs[1], obs[2]
                    )  # ... vel, acc, pedal
                    with data_lock:
                        throttle = vcu_output.torque
                        h1 = vcu_output.header

                        # the following 3 lines need to be deleted
                        throttle = vcu_lookup_table(
                            obs[2], obs[0]
                        )  # look up vcu table with pedal and speed  for throttle request
                else:
                    throttle = vcu_lookup_table(
                        obs[2], obs[0]
                    )  # look up vcu table with pedal and speed  for throttle request

                action = [throttle, 0]
                # print("action:{}".format(action[0]))
                # print("env.throttle:{}".format(env.pedal))
                # TODO l045a define episode done (time, distance, defined end event)
                obs, r, done, info = env.step(action)
                episode_done = done
                vcu_reward += r
                vcu_states.append(obs)

                # state has 20 [speed, acceleration, throttle] triplets, update policy (mu, sigma and update vcu)
                # update vcu calibration table every one second
                if (
                    timestep + 1
                ) % sequence_len == 0:  # sequence_len = 20; state.len == 20
                    vcu_states = tf.convert_to_tensor(
                        vcu_states
                    )  # state must have 20 (speed, acceleration, throttle) triples
                    vcu_states_history.append(vcu_states)
                    vcu_states = tf.expand_dims(vcu_states, 0)

                    # predict action probabilities and estimated future rewards
                    # from environment state
                    mu_sigma, critic_value = actorcritic_network(vcu_states)

                    vcu_critic_value_history.append(critic_value[0, 0])
                    mu_sigma_history.append(mu_sigma)

                    # sample action from action probability distribution
                    nn_mu, nn_sigma = tf.unstack(mu_sigma)
                    mvn = tfd.MultivariateNormalDiag(loc=nn_mu, scale_diag=nn_sigma)
                    vcu_action = mvn.sample()  # 17*21 =  357 actions
                    vcu_action_history.append(vcu_action)
                    # Here the loopup table with contrained output is part of the environemnt,
                    # clip is part of the environment to be learned
                    # action is not constrained!
                    vcu_calib_table = tf.reshape(
                        vcu_action, [vcu_calib_table_row, vcu_calib_table_col]
                    )
                    vcu_calib_table = tf.clip_by_value(
                        vcu_calib_table * vcu_calib_table_budget
                        + vcu_calib_table0,  # add changes to the default value
                        clip_value_min=0.0,
                        clip_value_max=1.0,
                    )

                    # Create a matplotlib 3d figure, //export and save in log
                    pd_data = pd.DataFrame(
                        vcu_calib_table.numpy(),
                        columns=np.linspace(0, 1.0, num=17),
                        index=np.linspace(0, 30, num=21),
                    )
                    df = pd_data.unstack().reset_index()
                    df.columns = ["pedal", "velocity", "throttle"]

                    fig = plt.figure()
                    ax = fig.gca(projection="3d")
                    surf = ax.plot_trisurf(
                        df["pedal"],
                        df["velocity"],
                        df["throttle"],
                        cmap=plt.cm.viridis,
                        linewidth=0.2,
                    )
                    fig.colorbar(surf, shrink=0.5, aspect=5)
                    ax.view_init(30, 135)
                    # throw figure to the visualization thread
                    figQueue.put(fig)
                    # plt.show()

                    # action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    # action_probs_history.append(tf.math.log(action_probs[0, action]))
                    # vcu_param_list = udp_sender.prepare_vcu_calibration_table(vcu_action.numpy())
                    # udp_sender.send_table(vcu_param_list)
                    if L045B:
                        # flashing calibration through xcp
                        # with flash_mutex:
                        vcu_act_list = vcu_calib_table.numpy().reshape(-1).tolist()
                        dataQueue.put(vcu_act_list)
                        print("update vcu calib table")
                        # send_float_array('TQD_trqTrqSetECO_MAP_v', vcu_calib_table.numpy().reshape(-1).tolist() )
                    else:
                        vcu_lookup_table = generate_lookup_table(
                            pedal_range, velocity_range, vcu_calib_table
                        )

                    # reward history
                    vcu_rewards_history.append(vcu_reward)
                    episode_reward += vcu_reward
                    vcu_reward = 0
                    vcu_states = []

                if done:
                    break

            # update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # todo calculate return
            # calculate expected value from rewards
            # - at each timestep what was the total reward received after that timestep
            # - rewards in the past are discounted by multiplying them with gamma
            # - these are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in vcu_rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # calculating loss values to update our network
            history = zip(
                vcu_action_history, mu_sigma_history, vcu_critic_value_history, returns
            )

            # back propagation
            (
                loss_all,
                act_losses_all,
                entropy_losses_all,
                critic_losses_all,
            ) = train_step(actorcritic_network, history, opt, tape)
            ckpt.step.assign_add(1)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss_sum", loss_all, step=episode_count)
            tf.summary.scalar("loss_act", act_losses_all, step=episode_count)
            tf.summary.scalar("loss_entropy", entropy_losses_all, step=episode_count)
            tf.summary.scalar("loss_critic", critic_losses_all, step=episode_count)
            tf.summary.scalar("episode reward", episode_reward, step=episode_count)

        output_template = "Episode {}, Loss all: {}, Act loss: {}, Entropy loss: {}, Critic loss: {}, Episode Reward: {}"
        print(
            output_template.format(
                episode_count + 1,
                loss_all,
                act_losses_all,
                entropy_losses_all,
                critic_losses_all,
                episode_reward,
            )
        )
        #
        #   # Reset metrics every epoch
        #   train_loss.reset_states()
        #   test_loss.reset_states()
        #   train_accuracy.reset_states()
        #   test_accuracy.reset_states()
        #   clear the loss and reward history
        vcu_states_history.clear()
        vcu_action_history.clear()
        vcu_rewards_history.clear()
        mu_sigma_history.clear()
        vcu_critic_value_history.clear()
        obs = env.reset()
        episode_reward = 0
        # log details
        # actorcritic_network.save_weights("./checkpoints/cp-{epoch:04d}.ckpt")
        # actorcritic_network.save("./checkpoints/cp-last.kpt")
        # ckp_moment = datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
        # last_model_save_path = f"./checkpoints/cp-{ckp_moment}-{episode_count}.ckpt"
        # actorcritic_network.save(last_model_save_path)

        # Checkpoint manager save model
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss {:1.2f}".format(loss_all.numpy()))

        episode_count += 1
        if episode_count % 1 == 0:
            template = "running reward: {:.2f} at episode {}"
            print("========================")
            print(template.format(running_reward, episode_count))

        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # if running_reward > 195:  # condition to consider the task solved
        #     print("solved at episode {}!".format(episode_count))
        #     break
    """
    ## visualizations
    in early stages of training:
    ![imgur](https://i.imgur.com/5gcs5kh.gif)
    
    in later stages of training:
    ![imgur](https://i.imgur.com/5ziizud.gif)
    """


if __name__ == "__main__":
    main()
