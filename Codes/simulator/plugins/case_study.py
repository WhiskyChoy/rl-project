""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from shapely.geometry import LineString

from bluesky import stack, traf as traffic, tools
from bluesky.proxy import PPOAgent
from bluesky.tools import geo

# For running on GPU
# from tensorflow.keras.backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)


# Initialization function of your plugin. Do not change the name of this
# function, as it is the way BlueSky recognises this file as a plugin.
num_ac: int
counter: int
max_ac: int
positions: np.ndarray
agent: PPOAgent
best_reward: float
num_success: int
success: bool
collision_number: int
ac_counter: int
route_queue: list
n_states: int
route_keeper: np.ndarray
previous_action: dict
last_observation: dict
observation: dict
num_success_train: list
num_collisions_train: list
choices: list
start: float


def init_plugin():
    global num_ac, counter, max_ac, positions, agent, best_reward, num_success, success, collision_number, \
        ac_counter, route_queue, n_states, route_keeper, previous_action, last_observation, \
        observation, num_success_train, num_collisions_train, choices, start

    num_success_train = []
    num_collisions_train = []

    num_success = []
    previous_action = {}
    last_observation = {}
    observation = {}
    collision_number = 0
    success = 0

    num_ac = 0
    max_ac = 30
    best_reward = -10000000
    ac_counter = 0
    n_states = 5
    route_keeper = np.zeros(max_ac, dtype=int)
    num_intruders = 20

    positions = np.load('./routes/case_study_route.npy')
    choices = [20, 25, 30]  # 4 minutes, 5 minutes, 6 minutes
    route_queue = random.choices(choices, k=positions.shape[0])

    agent = PPOAgent(n_states, 3, positions.shape[0], 100000, positions, num_intruders)
    counter = 0
    start = time.time()

    # Additional initialisation code
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'CASE_STUDY',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every time-step of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12.0,

        # The update function is called after trafficfic is updated. Use this if you
        # want to do things as a result of what happens in trafficfic. If you need to
        # something before trafficfic is updated please use pre-update.
        # 'reset': reset, #no need to auto reset
        'update': update}

    # If your plugin has a state, you will probably need a reset function to
    # clear the state in between simulations.
    # 'reset':         reset
    # }

    stack_functions = {}

    return config, stack_functions


# Periodic update functions that are called by the simulation. You can replace
# this by anything, so long as you communicate this in init_plugin
def set_route(i: int):
    global ac_counter, num_ac

    # print('set route called')

    lat, lon, g_lat, g_lon, h = positions[i]
    stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 251'.format(ac_counter, lat, lon, h))
    stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter, g_lat, g_lon))
    route_keeper[ac_counter] = i
    num_ac += 1
    ac_counter += 1


def update():
    """given a current state in the simulation, allow the agent to select an action.
     "" Then send the action to the bluesky command line ""
    """
    global num_ac, counter, max_ac, positions, agent, success, collision_number, ac_counter, route_queue, n_states, \
        route_keeper, previous_action, choices, start

    # print('update called')

    if ac_counter < max_ac:  # maybe spawn a/c based on time, not based on this update interval

        if ac_counter == 0:
            for i in range(len(positions)):
                set_route(i)
        else:
            for k in range(len(route_queue)):
                if counter == route_queue[k]:
                    set_route(k)

                    route_queue[k] = counter + random.choices(choices, k=1)[0]

                    if ac_counter == max_ac:
                        break

    store_terminal = np.zeros(len(traffic.id), dtype=int)
    for i in range(len(traffic.id)):
        label, type_ = agent.update(traffic, i, route_keeper)
        id_ = traffic.id[i]

        if label:
            stack.stack('DEL {}'.format(id_))
            num_ac -= 1
            if type_ == 1:
                collision_number += 1
            if type_ == 2:
                success += 1

            store_terminal[i] = 1

            agent.store(last_observation[id_], previous_action[id_], id_, type_)

            del last_observation[id_]

    if ac_counter == max_ac and num_ac == 0:
        # print(f'ac_counter:{ac_counter}, num_ac:{num_ac}')
        reset()
        return

    if num_ac == 0 and ac_counter != max_ac:
        return

    if not len(traffic.id) == 0:
        new_actions = {}
        n_ac = len(traffic.id)
        state = np.zeros((n_ac, n_states))

        id_sub = np.array(traffic.id)[store_terminal != 1]
        ind = np.array([int(x[2:]) for x in traffic.id])
        route = route_keeper[ind]

        state[:, 0] = traffic.lat
        state[:, 1] = traffic.lon
        state[:, 2] = traffic.tas
        state[:, 3] = route
        state[:, 4] = traffic.ax

        norm_state, norm_context = get_closest_ac(state, store_terminal)

        # if norm_state.shape[0] == 0:
        #     import ipdb; ipdb.set_trace()

        policy = agent.act(norm_state, norm_context)

        for j in range(len(id_sub)):
            id_ = id_sub[j]

            # This is for updating s, sp, ...
            if id_ not in last_observation.keys():
                last_observation[id_] = [norm_state[j], norm_context[j]]

            if id_ not in observation.keys() and id_ in previous_action.keys():
                observation[id_] = [norm_state[j], norm_context[j]]

                agent.store(last_observation[id_], previous_action[id_], id_)
                last_observation[id_] = observation[id_]

                del observation[id_]

            action = np.random.choice(agent.action_size, 1, p=policy[j].flatten())[0]
            speed = agent.speeds[action]
            index = traffic.id2idx(id_)

            if action == 1:  # hold
                speed = int(np.round((traffic.cas[index] / tools.geo.nm) * 3600))

            stack.stack('{} SPD {}'.format(id_, speed))
            new_actions[id_] = action

        previous_action = new_actions

    counter += 1


def reset():
    global best_reward
    global counter
    global num_ac
    global num_success
    global success
    global collision_number
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start

    # print('reset called')

    if (agent.episode_count + 1) % 5 == 0:
        agent.train()

    end = time.time()

    print(f'The time between two reset is: {round(end - start,2)} second(s)')
    goals_made = success

    num_success_train.append(success)
    num_collisions_train.append(collision_number)

    success = 0
    collision_number = 0

    counter = 0
    num_ac = 0
    ac_counter = 0

    route_queue = random.choices([20, 25, 30], k=positions.shape[0])

    previous_action = {}
    route_keeper = np.zeros(max_ac, dtype=int)
    last_observation = {}
    observation = {}

    t_success = np.array(num_success_train)
    t_coll = np.array(num_collisions_train)
    np.save('success_train_A.npy', t_success)
    np.save('collisions_train_A.npy', t_coll)

    if agent.episode_count > 150:
        df = pd.DataFrame(t_success)
        if float(df.rolling(150, 150).mean().max()) >= best_reward:
            agent.save(True, case_study='A')
            best_reward = float(df.rolling(150, 150).mean().max())

    agent.save(case_study='A')

    print("Episode: {} | Reward: {} | Best Reward: {}".format(agent.episode_count, goals_made, best_reward))

    agent.episode_count += 1

    if agent.episode_count == agent.numEpisodes:
        stack.stack('STOP')

    stack.stack('IC case_study.scn')

    start = time.time()


def get_closest_ac(state, store_terminal):
    global n_states, agent

    n_ac = traffic.lat.shape[0]
    norm_state = np.zeros((len(store_terminal[store_terminal != 1]), n_states))

    d = geo.latlondist_matrix(np.repeat(state[:, 0], n_ac), np.repeat(state[:, 1], n_ac), np.tile(state[:, 0], n_ac),
                              np.tile(state[:, 1], n_ac)).reshape(n_ac, n_ac)
    arg_sort = np.array(np.argsort(d, axis=1))

    total_closest_states = []

    max_agents = 1

    count = 0
    for i in range(d.shape[0]):
        r = int(state[i][3])
        lat, lon, g_lat, g_lon, h = agent.positions[r]
        if store_terminal[i] == 1:
            continue
        own_ship_obj = LineString([[state[i][1], state[i][0], 31000], [g_lon, g_lat, 31000]])

        norm_state[count, :] = agent.normalize(state[i], 'state', id_=traffic.id[i])
        closest_states = []
        count += 1

        route_count = 0

        intruder_count = 0

        for j in range(len(arg_sort[i])):

            index = int(arg_sort[i][j])

            if i == index:
                continue

            if store_terminal[index] == 1:
                continue

            route = int(state[index][3])

            if route == r and route_count == 2:
                continue

            if route == r:
                route_count += 1

            lat, lon, g_lat, g_lon, h = agent.positions[route]
            int_obj = LineString([[state[index, 1], state[index, 0], 31000], [g_lon, g_lat, 31000]])

            if not own_ship_obj.intersects(int_obj):
                continue

            if not (r in agent.intersection_distances.keys() and route in agent.intersection_distances[r].keys()) \
                    and route != r:
                continue

            if d[i, index] > 100:
                continue

            max_agents = max(max_agents, j)

            if len(closest_states) == 0:
                closest_states = np.array([traffic.lat[index], traffic.lon[index], traffic.tas[index], route,
                                           traffic.ax[index]])
                closest_states = agent.normalize(norm_state[count - 1], 'context', closest_states, state[i],
                                                 id_=traffic.id[index])
            else:
                closest_states = np.append(closest_states, closest_states, axis=1)

            intruder_count += 1

            if intruder_count == agent.num_intruders:
                break

        if len(closest_states) == 0:
            closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape([1, 1, 7])

        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:

            total_closest_states = np.append(
                tf.keras.preprocessing.sequence.pad_sequences(total_closest_states, agent.num_intruders,
                                                              dtype='float32'),
                tf.keras.preprocessing.sequence.pad_sequences(closest_states, agent.num_intruders, dtype='float32'),
                axis=0)

    if len(total_closest_states) == 0:
        total_closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape([1, agent.num_intruders, 7])

    return norm_state, total_closest_states
