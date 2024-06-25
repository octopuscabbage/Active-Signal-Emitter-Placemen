import logging
import random
from collections import defaultdict
from typing import List

import numpy as np
import xxhash
from igraph import Graph, plot
from numpy.random import multinomial
import matplotlib.pyplot as plt
import matplotlib
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm
from sample_sim.general_utils import round_sigfigs
import enum

class RolloutStrategy(enum.Enum):
    RANDON = 0,
    REWARD_WEIGHTED = 1,

class DiscreteWeightedBelief():
    def __init__(self, weights: List, items: List):
        self.weights = weights
        self.items = items

        assert len(self.items) == len(self.weights)

    def sample_item(self):
        return random.choices(population=self.items, weights=self.weights, k=1)[0]

    def increase_weight(self, item, weight_increase):
        try:
            self.weights[self.items.index(item)] += weight_increase
        except ValueError:
            self.items.append(item)
            self.weights.append(weight_increase)

        assert len(self.items) == len(self.weights)

    def add_item(self, item, weight):
        self.items.append(item)
        self.weights.append(weight)

        assert len(self.items) == len(self.weights)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return str(list(zip(self.items, self.weights)))

def fast_numpy_hash(xs):
    return xxhash.xxh32(xs).intdigest()

class HashableNumpyArray():
    def __init__(self,xs):
        self.xs = xs
    def __hash__(self):
        return fast_numpy_hash(np.ascontiguousarray(self.xs))
    def __str__(self):
        return str(self.xs.tostring())
    def __repr__(self):
        return repr(self.xs)
    def __eq__(self, other):
        if  isinstance(other,HashableNumpyArray):
            return (self.xs == other.xs).all()
        else:
            return False

class SamplingStateHistory():
    def __init__(self, current_position: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        self.current_position = current_position
        self.xs = xs
        self.ys = ys
    def __eq__(self, other):
        if not isinstance(other,SamplingStateHistory):
            return False
        else:
            return (self.current_position == other.current_position).all() and (self.xs == other.xs).all() and (self.ys == other.ys).all()
    def __hash__(self):
        h = xxhash.xxh32()
        h.update(self.current_position)
        h.update(np.ascontiguousarray(self.xs))
        h.update(np.ascontiguousarray(self.ys))
        return h.intdigest()

    def add_state(self, s):
        new_pos, x, y = s
        xs = np.vstack((self.xs, x))
        ys = np.concatenate((self.ys, y))
        return SamplingStateHistory(new_pos, xs, ys)

    def __str__(self):
        s = ""
        # if len(self.xs) > 3:
        #     s = "... \n"
        # for x, y in list(zip(self.xs, self.ys))[-3:]:
        #     s += f"{x} | {y} \n"
        s += f"pos {self.current_position} # {self.ys.shape[0]}"
        return s
    def __repr__(self):
        return str(self)

    def get_last_state(self):
        return (self.current_position, self.xs[:,-1],self.ys[-1])

class POMCPOW():
    def __init__(self, logger_name, next_action_fn, generator_fn, reward_fn, extra_data=None, max_depth=10, k_action=10,
                 alpha_action=0.1, k_observation=5, alpha_observation=1/15, q_init=0, exploration_weight_upper=1,exploration_weight_lower=1, gamma=0.99,
                 check_actions_repeated=True, check_observations_repeated=True, all_actions_fn=None):
        self.logger_name = logger_name
        self.num_tries = defaultdict(lambda: 0)  # Num times node is chosen

        self.belief_children = defaultdict(set)  # Observation to action children

        # Particle filters for POMCPOW
        self.belief_action_children = defaultdict(lambda: DiscreteWeightedBelief([], []))
        self.M = defaultdict(lambda: DiscreteWeightedBelief([], []))
        self.B = defaultdict(lambda: DiscreteWeightedBelief([], []))

        self.total_reward = defaultdict(lambda: q_init)
        self.max_depth = max_depth

        # double progressive widening stuff
        self.k_action = k_action
        self.alpha_action = alpha_action
        self.k_observation = k_observation
        self.alpha_observation = alpha_observation

        self.exploration_weight_upper = exploration_weight_upper
        self.exploration_weight_lower = exploration_weight_lower

        # User provided functions
        self.next_action = next_action_fn
        self.generator = generator_fn
        self.all_actions = all_actions_fn #only needed if you use a non-random rollout strategy

        self.gamma = gamma

        # Use this if you need to provide some extra data to a pomcp generator
        self.extra_data = extra_data

        self.reward_fn = reward_fn


        self.check_actions_repeated = check_actions_repeated
        self.check_observations_repeated = check_observations_repeated
        self.rewards_seen = []

    def plan(self, belief: DiscreteWeightedBelief, iterations):
        """
        Simulae the tree for a number of iterations from some initial belief
        :param belief:
        :param iterations:
        :return:
        """
        for i in tqdm(range(0, iterations),desc="POMCPOW Iterations", file=TqdmToLogger(logging.getLogger(self.logger_name))):
        #for i in range(0, iterations):
            state = belief.sample_item()
            self.simulate(state, state, self.max_depth)
        # TODO this uses the last sampled state
        return self.find_best_child(state, uct=False)

    def find_best_child(self, belief, uct=True):

        best_action = None
        best_value = float("-inf")
        for child_action in self.belief_children[belief]:
            if uct:
                value = self.uct(child_action, belief)
            else:
                value = self.total_reward[(belief, child_action)]
            if value > best_value:
                best_value = value
                best_action = child_action
        return best_action

    def action_progressive_widen(self, belief: SamplingStateHistory):
        if len(self.belief_children[belief]) <= self.k_action * self.num_tries[belief] ** self.alpha_action:
            a = self.next_action(belief, self.extra_data)
            if not self.check_actions_repeated or not a in self.belief_children[belief]:
                self.belief_children[belief].add(a)
        else:
            print("didn't sample action")
        return self.find_best_child(belief, uct=True)

    def uct(self, action, belief):
        action_visits = self.num_tries[(belief, action)]
        if action_visits == 0:
             return float("inf")
        Q = self.total_reward[(belief, action)]  
        parent_tries = float(self.num_tries[belief])
        current_tries = float(self.num_tries[(belief, action)])
        B = self.compute_exploration_weight() 
        E = B * np.sqrt(
            np.log(parent_tries) / (current_tries))

        U = Q + E        
        return U 

    def compute_exploration_weight(self):
        if self.exploration_weight_upper is None or self.exploration_weight_lower is None:
            self.exploration_weight_upper = max(self.rewards_seen)
            self.exploration_weight_lower = min(self.rewards_seen) - 10**-5
        else:
            self.exploration_weight_upper = max(self.exploration_weight_upper,max(self.rewards_seen))
            self.exploration_weight_lower = min(self.exploration_weight_lower,min(self.rewards_seen))
        return  np.sqrt(2) #* (self.exploration_weight_upper - self.exploration_weight_lower)


    def simulate(self, state, belief: SamplingStateHistory, depth):
        if depth == 0:
            return 0
        action = self.action_progressive_widen(belief)
        next_state, observation, reward, observation_probability = self.generator(belief, action, self.extra_data)
        self.rewards_seen.append(reward)
        self.compute_exploration_weight()
        reward = (reward - self.exploration_weight_lower) / (self.exploration_weight_upper - self.exploration_weight_lower)

        #Need to do this here because we will automatically add it in a second
        observation_not_in_belief_action_children = observation not in self.belief_action_children[(belief, action)].items

        if len(self.belief_action_children[(belief, action)]) <= self.k_observation * self.num_tries[
            (belief, action)] ** self.alpha_observation:
            self.belief_action_children[(belief, action)].increase_weight(observation, 1)
            #print(f"Adding new observation L{len(self.belief_action_children[(belief,action)])} N: {self.num_tries[(belief,action)]}, A: {self.k_observation * self.num_tries[(belief,action)] ** self.alpha_observation}")
        else:
            observation = self.belief_action_children[(belief, action)].sample_item()
            #print("Not adding observation")

        self.B[(belief, action, observation)].increase_weight(next_state, observation_probability) #Either adds a new state or increases the likliehood

        if observation_not_in_belief_action_children:
            #self.belief_action_children[(belief, action)].add_item(observation, 1) #We don't need to do this because we already increased the weight
            total = reward + self.gamma * self.rollout(next_state, belief.add_state(next_state), depth - 1)
        else:
            next_state = self.B[(belief, action, observation)].sample_item()
            reward = self.reward_fn(belief, action, next_state,self.extra_data)
            self.rewards_seen.append(reward)
            self.compute_exploration_weight()
            reward = (reward - self.exploration_weight_lower) / (self.exploration_weight_upper - self.exploration_weight_lower)

            total = reward + self.gamma * self.simulate(next_state, belief.add_state(next_state), depth - 1)

        self.num_tries[belief] += 1
        self.num_tries[(belief, action)] += 1
         
        self.total_reward[(belief, action)] += (total - self.total_reward[(belief, action)]) / self.num_tries[
            (belief, action)]

        return total

    def rollout(self, state, belief: SamplingStateHistory, depth):
        if depth == 0:
            return 0
        
        if self.extra_data.rollout_strategy == RolloutStrategy.RANDON:
            action = self.next_action(belief, self.extra_data)
        elif self.extra_data.rollout_strategy == RolloutStrategy.REWARD_WEIGHTED:
            actions = self.all_actions(belief,self.extra_data)
            rewards = []
            for action in actions:
                _, _, reward, _ = self.generator(belief, action, self.extra_data)
                rewards.append(reward)
            action = random.choices(actions,weights=rewards)[0]
        else:
            raise Exception(f"Unknown Rollout Strategy {self.extra_data.rollout_strategy}")
        sample_state, _, reward, _ = self.generator(belief, action, self.extra_data)
        self.rewards_seen.append(reward)
        self.compute_exploration_weight()
        reward = (reward - self.exploration_weight_lower) / (self.exploration_weight_upper - self.exploration_weight_lower)
        discounted_reward = reward + self.gamma * self.rollout(state, belief.add_state(sample_state), depth - 1)
        return discounted_reward

    def to_string(self, belief, depth=1, max_depth=3):
        if depth > max_depth:
            return ""

        s = ""
        beliefs = []
        if isinstance(belief, DiscreteWeightedBelief):
            for cur_belief in belief.items:
                if not isinstance(cur_belief, tuple):
                    cur_belief = (cur_belief,)
                    beliefs.append(cur_belief)

        else:
            beliefs = [belief]
        for cur_belief in beliefs:
            for action in self.belief_children[cur_belief]:
                if self.num_tries[(cur_belief, action)] == 0:
                    continue
                s += "|" * depth + f"B: {cur_belief} {action}, Q: {round(self.total_reward[(cur_belief, action)],2)} N: {self.num_tries[(cur_belief, action)]} \n"
                for observation in self.belief_action_children[(cur_belief, action)].items:
                    s += "-"*depth + f"{observation} \n"
                    for next_state in self.B[(cur_belief, action, observation)].items:
                        s += "*" * depth + f"{next_state}\n"
                        s += self.to_string(cur_belief.add_state(next_state), depth + 1)

        return s

    def draw_tree_igraph(self, initial_belief:SamplingStateHistory, max_depth=3, bounds=(2400, 1200),name="",fignum=1):
        label_size = 8
        g = Graph(directed=True)
        #queue = list(map(lambda x: (None, (x,)), initial_belief.items))
        queue = [[0,None,initial_belief]]
        while queue:
            depth, parent_index, belief = queue.pop()
            if parent_index is None:
                root_node = g.add_vertex(name=f"{belief}")
                g.vs[root_node.index][
                    "label"] = f"{belief} \n  N {self.num_tries[(belief)]}"
                g.vs[root_node.index]["size"] = .5
                g.vs[root_node.index]["color"] = "blue"
                g.vs[root_node.index]["label_size"] = label_size


                parent_index =root_node.index
            for action in self.belief_children[belief]:
                # if belief == initial_belief :
                #     if self.num_tries[(belief,action)] == 0:
                #         print(f"Never tried {action}")
                #     else:
                #         print(f"tried {action}")
                if self.num_tries[(belief, action)] == 0:
                    continue

                if len(set(self.belief_children[belief])) != len(self.belief_children[belief]):
                    raise Exception()
                belief_node = g.add_vertex(name=f"{belief} {action}")
                g.vs[belief_node.index]["label"] = f"{belief} \n {action} \n Q {round(self.total_reward[(belief, action)],2)} N {self.num_tries[(belief, action)]} U: {round(self.uct(action,belief),2)}"

                g.vs[belief_node.index]["size"] = .5
                g.vs[belief_node.index]["color"] = "red"
                g.vs[belief_node.index]["label_size"] = label_size

                #if parent_index is not None:
                g.add_edges([[parent_index, belief_node.index]])
                for observation in self.belief_action_children[(belief, action)].items:
                    pos, value = observation
                    obs_format = f"C {len(self.B[(belief,action,observation)])} \n"
                    for cur_pos, cur_value in zip(pos.xs,value.xs):
                        obs_format += f"{(list(cur_pos))} | {round_sigfigs(cur_value,2)} \n"
                    #obs_format += f"{list(pos.xs[:,-1])} | {round_sigfigs(value.xs[-1],2)}"
                    obs_node = g.add_vertex(obs_format)
                    g.vs[obs_node.index]["label"] = obs_format
                    g.vs[obs_node.index]["color"] = "green"
                    g.vs[obs_node.index]["size"] = .3
                    g.vs[obs_node.index]["label_size"] = label_size

                    g.add_edges([[belief_node.index, obs_node.index]])
                    if depth+1 < max_depth:
                        next_state_set = set()
                        for next_state in self.B[(belief, action, observation)].items:
                            hashable_next_state = tuple(map(HashableNumpyArray,next_state))
                            if hashable_next_state not in next_state_set:
                                next_state_set.add(hashable_next_state)
                                queue.append((depth+1, obs_node.index, belief.add_state(next_state)))
        #print(g)
        layout = g.layout("reingold_tilford_circular")
        #plt.ion()
        plt.figure(fignum,figsize=(10,8))
        plt.clf()
        #ax = plt.gca()
        plot(g, target=plt.gca(), layout=layout, bbox=bounds,margin=150)



if __name__ == "__main__":
    S = {"Here": 0, "There": 1}
    Sinv = {v: k for k, v in S.items()}
    A = {"Left": 0, "Right": 1}
    O = {"Red": 0, "Yellow": 1}
    Oinv = {v: k for k, v in O.items()}

    # new_s, s, a
    a = np.zeros((2, 2, 2))
    a[0, 0, 0] = 0.9
    a[1, 0, 0] = 0.1
    a[0, 1, 0] = 0.35
    a[1, 1, 0] = 0.65
    a[0, 0, 1] = 0.4
    a[1, 0, 1] = 0.6
    a[0, 1, 1] = 0.8
    a[1, 1, 1] = 0.2

    # obs, s, a
    b = np.zeros((2, 2, 2))
    b[0, 0, 0] = 0.8
    b[1, 0, 0] = 0.2
    b[0, 1, 0] = 0.3
    b[1, 1, 0] = 0.7
    b[0, 0, 1] = 0.4
    b[1, 0, 1] = 0.6
    b[0, 1, 1] = 0.5
    b[1, 1, 1] = 0.5

    # s, a
    # rewards
    # s,action
    r = np.zeros((2, 2))
    r[0, 0] = 1
    r[0, 1] = 50
    r[1, 0] = 1
    r[1, 1] = 2


    # define a black box generator
    def Generator(s, act, extra_data):
        ss = multinomial(1, a[:, S[s], A[act]])
        ss = np.nonzero(ss)[0][0]
        o = multinomial(1, b[:, ss, A[act]])
        o = np.nonzero(o)[0][0]
        rw = r[S[s], A[act]]
        return Sinv[ss], Oinv[o], rw, 0.5


    def action_fn(belief, extra_data):
        return random.choice(["Left", "Right"])


    def reward_fn(state, action, next_state):
        return r[S[state], A[action]]


    planner = POMCPOW(generator_fn=Generator, next_action_fn=action_fn, reward_fn=reward_fn)
    belief = DiscreteWeightedBelief([1], ["Here"])
    planner.plan(belief, 1000)
    # print(planner.to_string(belief,max_depth=1))
    planner.draw_tree_igraph(belief, max_depth=3, bounds=(1000, 400))
