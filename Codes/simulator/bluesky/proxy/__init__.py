import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from bluesky.tools import geo
from operator import itemgetter
from shapely.geometry import LineString
import numba as nb

LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-4
HIDDEN_SIZE = 32


@nb.njit()
def discount(r, discounted_r, accumulate_r):
    for t in range(len(r) - 1, -1, -1):
        accumulate_r = r[t] + accumulate_r * 0.99
        discounted_r[t] = accumulate_r
    return discounted_r


def dist_goal(states, trafficfic, i):
    o_lat, o_lon = states
    i_lat, i_lon = trafficfic.ap.route[i].wplat[0], trafficfic.ap.route[i].wplon[0]
    dist = geo.latlondist(o_lat, o_lon, i_lat, i_lon) / geo.nm
    return dist


def get_closest_ac_distance(self, state, traffic, route_keeper):
    o_lat, o_lon, id_ = state[:3]
    index = int(id_[2:])
    rte = int(route_keeper[index])
    lat, lon, g_lat, g_lon, h = self.positions[rte]
    size = traffic.lat.shape[0]
    index = np.arange(size).reshape(-1, 1)
    ownship_obj = LineString([[o_lon, o_lat, 31000], [g_lon, g_lat, 31000]])
    d = geo.latlondist_matrix(np.repeat(o_lat, size), np.repeat(o_lon, size), traffic.lat, traffic.lon)
    d = d.reshape(-1, 1)

    dist = np.concatenate([d, index], axis=1)

    dist = sorted(np.array(dist), key=itemgetter(0))[1:]
    if len(dist) > 0:
        for i in range(len(dist)):

            index = int(dist[i][1])
            id_ = traffic.id[index]
            index_route = int(id_[2:])

            rte_int = route_keeper[index_route]
            lat, lon, g_lat, g_lon, h = self.positions[rte_int]
            int_obj = LineString([[traffic.lon[index], traffic.lat[index], 31000], [g_lon, g_lat, 31000]])

            if not ownship_obj.intersects(int_obj):
                continue

            if not (rte in self.intersection_distances.keys() and rte in self.intersection_distances[rte].keys()) \
                    and rte_int != rte:
                continue

            if dist[i][0] > 100:
                continue

            return dist[i][0]
    else:
        return np.inf

    return np.inf


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -k.mean(k.minimum(r * advantage, k.clip(r, min_value=1 - LOSS_CLIPPING,
                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                prob * k.log(prob + 1e-10)))

    return loss


class PPOAgent:
    def __init__(self, state_size, action_size, num_routes, num_episodes, positions, num_intruders):

        self.state_size = state_size
        self.action_size = action_size
        self.positions = positions
        self.gamma = 0.99  # discount rate
        self.numEpisodes = num_episodes
        self.max_time = 500
        self.num_intruders = num_intruders

        self.episode_count = 0
        self.speeds = np.array([156, 0, 346])
        self.max_agents = 0
        self.num_routes = num_routes
        self.experience = {}
        self.dist_close = {}
        self.dist_goal = {}
        self.tas_max = 253.39054470774
        self.tas_min = 118.54804803287088
        self.lr = 0.0001
        self.value_size = 1
        self.max_d = 0
        self.intersections = {}
        self.intersection_distances = {}
        self.route_distances = []
        self.conflict_routes = {}

        self.get_route_distances()

        self.model_check = []
        self.model = self._build_ppo()
        self.count = 0

    def get_route_distances(self):

        i = 0

        for i in range(len(self.positions)):
            o_lat, o_lon, g_lat, g_lon, h = self.positions[i]
            _, d = geo.qdrdist(o_lat, o_lon, g_lat, g_lon)
            self.route_distances.append(d)
            own_obj = LineString([[o_lon, o_lat, 31000], [g_lon, g_lat, 31000]])
            self.conflict_routes[i] = []
            for j in range(len(self.positions)):
                if i == j:
                    continue
                o_lat, o_lon, g_lat, g_lon, h = self.positions[j]
                other_obj = LineString([[o_lon, o_lat, 31000], [g_lon, g_lat, 31000]])
                self.conflict_routes[i].append(j)
                if own_obj.intersects(other_obj):
                    intersect = own_obj.intersection(other_obj)
                    try:
                        i_lon, i_lat, alt = list(list(intersect.boundary[0].coords)[0])
                    except IndexError:
                        i_lon, i_lat, alt = list(list(intersect.coords)[0])

                    try:
                        self.intersections[i].append([j, [i_lat, i_lon]])
                    except KeyError:
                        self.intersections[i] = [[j, [i_lat, i_lon]]]

        for route in self.intersections.keys():
            o_lat, o_lon, g_lat, g_lon, h = self.positions[i]

            for intersections in self.intersections[route]:
                conflict_route, location = intersections
                i_lat, i_lon = location
                _, d = geo.qdrdist(i_lat, i_lon, g_lat, g_lon)
                try:
                    self.intersection_distances[route][conflict_route] = d
                except KeyError:
                    self.intersection_distances[route] = {conflict_route: d}

        self.max_d = max(self.route_distances)

    def normalize(self, value, what, context=None, state=None, id_=None):

        if what == 'spd':

            if value > self.tas_max:
                self.tas_max = value

            if value < self.tas_min:
                self.tas_min = value
            return (value - self.tas_min) / (self.tas_max - self.tas_min)

        if what == 'rt':
            return value / (self.num_routes - 1)

        if what == 'state':
            d_goal = self.dist_goal[id_] / self.max_d
            spd = self.normalize(value[2], 'spd')
            rt = self.normalize(value[3], 'rt')
            acc = value[4] + 0.5
            norm_array = np.array([d_goal, spd, rt, acc, 3 / self.max_d])

            return norm_array

        if what == 'context':

            rt_own = int(state[3])
            d_goal = self.dist_goal[id_] / self.max_d
            spd = self.normalize(context[2], 'spd')
            rt = self.normalize(context[3], 'rt')
            acc = context[4] + 0.5
            rt_int = int(context[3])

            if rt_own == rt_int:
                dist_away = abs(value[0] - d_goal)
                dist_own_intersection = 0
                dist_int_intersection = 0  #

            else:
                dist_own_intersection = abs(self.intersection_distances[rt_own][rt_int] / self.max_d - value[0])
                dist_int_intersection = abs(self.intersection_distances[rt_int][rt_own] / self.max_d - d_goal)
                d = geo.latlondist(state[0], state[1], context[0], context[1]) / geo.nm
                dist_away = d / self.max_d

            context_arr = np.array([d_goal, spd, rt, acc, dist_away, dist_own_intersection, dist_int_intersection])

            return context_arr.reshape([1, 1, 7])

    def _build_ppo(self):

        input_ = tf.keras.layers.Input(shape=(self.state_size,), name='states')

        context = tf.keras.layers.Input(shape=(self.num_intruders, 7), name='context')
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,), name='empty')

        advantage = tf.keras.layers.Input(shape=(1,), name='A')
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,), name='old_pred')

        flatten_context = tf.keras.layers.Flatten()(context)
        # encoding other_state into 32 values
        h1_int = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(flatten_context)
        # now combine them
        combined = tf.keras.layers.concatenate([input_, h1_int], axis=1)

        h2 = tf.keras.layers.Dense(256, activation='relu')(combined)
        h3 = tf.keras.layers.Dense(256, activation='relu')(h2)

        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(h3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)

        # now I need to apply activation
        policy_out = tf.keras.layers.Activation('softmax', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=[input_, context, empty, advantage, old_prediction],
                                      outputs=[policy_out, value_out])

        self.predictor = tf.keras.models.Model(inputs=[input_, context, empty], outputs=[policy_out, value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out': proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=old_prediction), 'value_out': 'mse'})

        print(model.summary())

        return model

    def store(self, state, action, id_, type_=0):
        reward = 0
        done = False

        if type_ == 0:

            dist = self.dist_close[id_]

            if 10 > dist > 3:
                reward = -0.1 + 0.05 * (dist / 10)

        if type_ == 1:
            reward = -1
            done = True

        if type_ == 2:
            reward = 0
            done = True

        state, context = state
        state = state.reshape((1, self.state_size))
        context = context.reshape((1, -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]

        self.max_agents = max(self.max_agents, context.shape[1])

        if id_ not in self.experience.keys():
            self.experience[id_] = {}

        try:
            self.experience[id_]['state'] = np.append(self.experience[id_]['state'], state, axis=0)

            if self.max_agents > self.experience[id_]['context'].shape[1]:
                self.experience[id_]['context'] = np.append(
                    tf.keras.preprocessing.sequence.pad_sequences(self.experience[id_]['context'], self.max_agents,
                                                                  dtype='float32'), context, axis=0)
            else:
                self.experience[id_]['context'] = np.append(self.experience[id_]['context'],
                                                            tf.keras.preprocessing.sequence.
                                                            pad_sequences(context,
                                                                          self.max_agents,
                                                                          dtype='float32'),
                                                            axis=0)

            self.experience[id_]['action'] = np.append(self.experience[id_]['action'], action)
            self.experience[id_]['reward'] = np.append(self.experience[id_]['reward'], reward)
            self.experience[id_]['done'] = np.append(self.experience[id_]['done'], done)

        except KeyError:
            self.experience[id_]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[id_]['context'] = tf.keras.preprocessing.sequence.pad_sequences(context,
                                                                                                self.max_agents,
                                                                                                dtype='float32')
            else:
                self.experience[id_]['context'] = context

            self.experience[id_]['action'] = [action]
            self.experience[id_]['reward'] = [reward]
            self.experience[id_]['done'] = [done]

    def train(self):

        """Grab samples from batch to train the network"""
        # print('train')

        total_state = []
        total_reward = []
        total_a = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in self.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            state = transitions['state']  # .reshape((episode_length,self.state_size))
            context = transitions['context']
            reward = transitions['reward']
            action = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward, discounted_r, cumul_r)
            policy, values = self.predictor.predict(
                {'states': state, 'context': context, 'empty': np.zeros((len(state), HIDDEN_SIZE))}, batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index, action] = 1
            a_val = discounted_rewards - values[:, 0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == self.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(context, self.max_agents,
                                                                                  dtype='float32')
                total_reward = discounted_rewards
                total_a = a_val
                total_advantage = advantages
                total_policy = policy

            else:
                total_state = np.append(total_state, state, axis=0)
                if context.shape[1] == self.max_agents:
                    total_context = np.append(total_context, context, axis=0)
                else:
                    total_context = np.append(total_context,
                                              tf.keras.preprocessing.sequence.pad_sequences(context, self.max_agents,
                                                                                            dtype='float32'), axis=0)
                total_reward = np.append(total_reward, discounted_rewards, axis=0)
                total_a = np.append(total_a, a_val, axis=0)
                total_advantage = np.append(total_advantage, advantages, axis=0)
                total_policy = np.append(total_policy, policy, axis=0)

        total_a = (total_a - total_a.mean()) / (total_a.std() + 1e-8)
        self.model.fit({'states': total_state, 'context': total_context, 'empty': np.zeros((total_length, HIDDEN_SIZE)),
                        'A': total_a, 'old_pred': total_policy},
                       {'policy_out': total_advantage, 'value_out': total_reward}, shuffle=True,
                       batch_size=total_state.shape[0], epochs=8, verbose=0)

        self.max_agents = 0
        self.experience = {}

    def load(self, name):
        print('Loading weights...')
        self.model.load_weights(name)
        print('Successfully loaded model weights from {}'.format(name))

    def save(self, best=False, case_study='A'):

        if best:
            self.model.save_weights('best_model_{}.h5'.format(case_study))
        else:
            self.model.save_weights('model_{}.h5'.format(case_study))

    # action implementation for the agent
    def act(self, state, context):

        context = context.reshape((state.shape[0], -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(context, self.num_intruders, dtype='float32')

        policy, value = self.predictor.predict(
            {'states': state, 'context': context, 'empty': np.zeros((state.shape[0], HIDDEN_SIZE))},
            batch_size=state.shape[0])

        return policy

    def update(self, traffic, index, route_keeper):
        """calulate reward and determine if terminal or not"""
        label = 0
        type_ = 0
        dist = get_closest_ac_distance(self, [traffic.lat[index], traffic.lon[index], traffic.id[index]], traffic,
                                       route_keeper)
        if dist < 3:
            label = True
            type_ = 1

        self.dist_close[traffic.id[index]] = dist

        d_goal = dist_goal([traffic.lat[index], traffic.lon[index]], traffic, index)

        if d_goal < 5 and label == 0:
            label = True
            type_ = 2

        self.dist_goal[traffic.id[index]] = d_goal

        return label, type_
