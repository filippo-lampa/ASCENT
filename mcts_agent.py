import numpy as np

from copy import deepcopy
from math import *
import random

import torch
from sympy.physics.units import current
from torch import nn

from replay_buffer import ReplayBuffer
from networks.utility import inference, training_model, observation_to_tensor
from utils.consts import mutant_operators_list

class AsymmetricLoss(nn.Module):
    def __init__(self, alpha):  # alpha > 1 penalizes underestimation more
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        errors = targets - predictions

        loss = torch.where(errors < 0,  # underestimation case
                           self.alpha * errors ** 2,
                           errors ** 2)  # normal penalty for overestimation

        return loss.mean()  # return average loss


class MCTSAgent:

    def __init__(self, policy_nn=None, value_nn=None, observation_nn=None, tests=None, kills_matrix=None,
                 sut_name=None, number_of_mutants=None, buffer_size=None, batch_size=None, update_delta=None,
                 observation_update_delta=None, observation_buffer_size=None, rollout_after=None,
                 asymmetric_loss_alpha=None, c_parameter=None, value_network_learning_rate=None,
                 policy_network_learning_rate=None, observation_network_learning_rate=None):

        # Set the device for PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using device: ", device)
            print(torch.cuda.get_device_name(0))

        # Init Neural-MCTS parameters
        self.ROLLOUT_AFTER = rollout_after
        self.BUFFER_SIZE = number_of_mutants if buffer_size is None else buffer_size
        self.BATCH_SIZE = batch_size
        self.UPDATE_DELTA = update_delta
        self.OBSERVATION_UPDATE_DELTA = observation_update_delta
        self.OBSERVATION_BUFFER_SIZE = observation_buffer_size
        self.OBSERVATION_KILL_BUFFER = []
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE)

        # Init networks
        self.policy_net = policy_nn.to(device)
        self.observation_nn = observation_nn.to(device)
        self.value_net = value_nn.to(device)
        self.asymmetric_loss_alpha = asymmetric_loss_alpha
        self.observation_nn_opt = torch.optim.Adam(self.observation_nn.parameters(), lr=observation_network_learning_rate)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=value_network_learning_rate)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=policy_network_learning_rate)
        self.observation_loss_function = torch.nn.BCEWithLogitsLoss()
        self.value_loss_function = AsymmetricLoss(self.asymmetric_loss_alpha)
        self.policy_loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
        self.loss_o = None # current loss of the observation network. used to track the training of the network

        # Others
        self.kills_matrix = kills_matrix
        self.tests = tests
        self.sut_name = sut_name
        self.num_actions = len(self.tests)
        self.max_reward = len(tests) # the maximum reward of the current episode to scale the values
        self.kills_ranking = {test: 0 for test in range(len(self.tests))} # kills ranking of the tests, used as heuristic until we start relying on the networks
        self.done = False # Checks if the episode is done (the mutant is killed or we run out of tests)
        self.current_sut_tests_execution_time = 0

        # Mutant-related stuff
        self.mutant_number = None  # number of the current mutant in the prioritization execution
        self.mutant = None
        self.number_of_tests_executed = 0
        self.number_of_tests_executed_on_killable_mutants = 0
        self.mutant_not_killable = None

        # MCTS stuff
        self.c = c_parameter
        self.tree = None
        self.root_nodes = []


    def init_tree(self):
        for i in range(len(self.tests)):
            initial_state = self.State([i], mutant_operators_list.index(self.mutant["operator"]), i, self.kills_matrix[self.mutant["operator"]][self.tests[i]["test_id"]])
            self.tree = self.Node(False, False, None, initial_state, i, self)
            self.root_nodes.append(self.tree)

    class State:
        def __init__(self, test_sequence: list[int], mutant_operator: str, test_index: int = 0, num_of_mutants_same_operator_killed: int = 0):
            self.test_sequence = test_sequence
            self.mutant_operator = mutant_operator
            self.test_index = test_index
            self.num_of_mutants_same_operator_killed = num_of_mutants_same_operator_killed

    class Node:
        def __init__(self, done, killed, parent, observation, action_index, mcts_agent):
            self.mcts_agent = mcts_agent
            self.children = {}
            self.T = 0 # sum of rewards
            self.N = 0 # number of visits
            self.observation = observation # state of the prioritization
            self.done = done # if the node is terminal. could be because we ran out of tests or because the mutant is killed
            self.killed = killed # if the mutant is killed
            self.parent = parent
            self.backup_parent = parent
            self.action_index = action_index # action index that leads to this node
            self.nn_v = 0 # value from the value network
            self.nn_p = [0] * self.mcts_agent.num_actions # priors from the policy network


        def getUCBscore(self):
            if self.N == 0:
                return float('inf')

            # We need the parent node of the current node
            top_node = self
            if top_node.parent:
                top_node = top_node.parent

            value_score = (self.T / self.N)

            prior_score = 0
            if self.mcts_agent.mutant_number >= self.mcts_agent.ROLLOUT_AFTER:
                prior_score = self.mcts_agent.c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)

            return value_score + prior_score


        def detach_parent(self):
            del self.parent
            self.parent = None


        def get_available_actions(self):
            return [i for i in range(len(self.mcts_agent.tests)) if i not in self.observation.test_sequence]


        def create_child(self):
            '''
            We create a copy of the current node enviroment and apply it to the new child, i.e. the most promising action
            '''

            if self.done:
                return

            possible_actions = self.get_available_actions()

            if len(possible_actions) == 0:
                print("no possible actions")
                return

            action = random.choice(possible_actions)

            # selective widening

            if self.mcts_agent.mutant_number < self.mcts_agent.ROLLOUT_AFTER:
                # get the most promising action according to the kills ranking
                for i in range(len(self.mcts_agent.kills_ranking)):
                    if list(self.mcts_agent.kills_ranking.keys())[i] in possible_actions:
                        action = list(self.mcts_agent.kills_ranking.keys())[i]
                        break
            else:
                # choose the action index that is in the possible actions and has the highest policy score
                action = max(possible_actions, key=lambda x: self.nn_p[x])

            env_copy = deepcopy(self.observation)
            env_copy.test_sequence.append(action)

            mutant_killed_prediction = inference(observation_to_tensor(env_copy, action, self.mcts_agent.num_actions), self.mcts_agent.observation_nn)
            placeholder_observation = self.mcts_agent.State(env_copy.test_sequence, env_copy.mutant_operator, action, env_copy.num_of_mutants_same_operator_killed)
            killed = 1 if mutant_killed_prediction > 1 else 0
            done = True if len(env_copy.test_sequence) == len(self.mcts_agent.tests) else False

            self.children[action] = type(self)(done, killed, self, placeholder_observation, action, self.mcts_agent)
            self.children[action].nn_v, self.children[action].nn_p = self.children[action].rollout()

            if self.mcts_agent.mutant_number >= self.mcts_agent.ROLLOUT_AFTER:
                self.children[action].T += self.children[action].nn_v


        def rollout(self):
            if self.done:
                return 0, None
            else:

                obs = observation_to_tensor(self.observation, total_number_of_tests=self.mcts_agent.num_actions)

                v = inference(obs, self.mcts_agent.value_net)
                p = inference(obs, self.mcts_agent.policy_net)

                return v if v > 0 else 0, p


        def next(self):

            if self.done:
                raise ValueError("episode has ended")

            if not self.children:
                raise ValueError('no children found and episode hasn\'t ended')

            for child in self.children.values():
                child.parent = child.backup_parent

            probs = [0] * len(self.children)
            max_U = max(c.getUCBscore() for c in self.children.values())

            if max_U == float('inf'):
                probs = [1 if c.getUCBscore() == float('inf') else 0 for c in self.children.values()]
            elif max_U == 0:
                probs = [1 / len(self.children) for c in self.children.values()]
            else:
                for index, node in enumerate(self.children.values()):
                    probs[index] = node.getUCBscore() / max_U

            probs = np.array(probs)
            probs = probs / probs.sum()

            #the next children is the one with the highest UCB score (the one with the highest probability)
            next_child = list(self.children.items())[np.argmax(probs)][1]

            #mask probabilities of all other actions to 0
            masked_probs = [0] * self.mcts_agent.num_actions
            for index, child in enumerate(self.children.values()):
                masked_probs[child.action_index] = probs[index]

            #fit the chosen node to the current mutant
            next_child.observation = self.mcts_agent.State(next_child.observation.test_sequence,
                                        mutant_operators_list.index(self.mcts_agent.mutant["operator"]),
                                        next_child.action_index, self.mcts_agent.kills_matrix[self.mcts_agent.mutant["operator"]][self.mcts_agent.tests[next_child.action_index]["test_id"]])

            current.nn_p = masked_probs

            return next_child, next_child.action_index, next_child.observation, masked_probs, self.observation


    def policy_player_mcts(self, mytree):

        mytree.N += 1 # We increment the number of visits of the current node

        if self.mutant_number >= self.ROLLOUT_AFTER:

            # perform a rollout to fit the current node to the current mutant, so that the next ucb score will be calculated
            # with the current mutant in mind
            nn_v, nn_p = mytree.rollout()
            mytree.nn_v = nn_v
            mytree.nn_p = nn_p
            mytree.T += nn_v

        mytree.create_child()

        next_tree, next_action, obs, p, p_obs = mytree.next()

        # we detach the current node and returning the sub-tree that starts from the node rooted at the choosen action
        next_tree.detach_parent()

        return next_tree, next_action, obs, p, p_obs


    def execute_test_on_mutant(self, test, mutant):
        """
        Execute the test against the mutant and return the outcome.
        """

        if not self.mutant_not_killable:
            self.number_of_tests_executed_on_killable_mutants += 1

        self.number_of_tests_executed += 1

        # each mutant, among the other information, has two parameters: killing and nonkilling tests.
        # right now, for testing purposes, we mock the execution of the test against the mutant by checking if the test
        # is in the killing tests of the mutant. If it is, the mutant is killed, otherwise it is not.
        outcome = 0
        if len(mutant["testResults"]) > 0:
            if "\\" in list(mutant["testResults"].keys())[0] and "/" in test["test_file_path"]:
                test_relative_path = test["test_file_path"].split(self.sut_name)[1].replace("/", "\\")
            elif "/" in list(mutant["testResults"].keys())[0] and "\\" in test["test_file_path"]:
                test_relative_path = test["test_file_path"].split(self.sut_name)[1].replace("\\", "/")
            else:
                test_relative_path = test["test_file_path"].split(self.sut_name)[1]

            duration_found = False

            for killing_test_method in mutant["testResults"][test_relative_path]["failed"]:
                if killing_test_method["title"] == test["test_method_name"]:
                    self.current_sut_tests_execution_time += killing_test_method["duration"]
                    duration_found = True
                    outcome = 1
                    break

            if not duration_found:
                for non_killing_test_method in mutant["testResults"][test_relative_path]["passed"]:
                    if non_killing_test_method["title"] == test["test_method_name"]:
                        self.current_sut_tests_execution_time += non_killing_test_method["duration"]
                        break

        # Update the kills ranking
        if outcome == 1:
            self.kills_ranking[self.tests.index(test)] = self.kills_ranking[self.tests.index(test)] + 1 if self.tests.index(test) in self.kills_ranking.keys() else 1
            self.kills_ranking = dict(sorted(self.kills_ranking.items(), key=lambda item: item[1], reverse=True))

        return outcome


    def take_step(self, action, env):
        """
        Perform all necessary actions on the environment after selecting an action.
        """

        # Execute the test against the mutant
        killed = self.execute_test_on_mutant(self.tests[action], self.mutant)
        self.kills_matrix[self.mutant["operator"]][self.tests[action]["test_id"]] += killed

        # train the observation network
        if (killed == 0 and random.random() < 0.1) or killed == 1:
            self.OBSERVATION_KILL_BUFFER.append([(env, action), killed])
            self.OBSERVATION_KILL_BUFFER = self.OBSERVATION_KILL_BUFFER[-self.OBSERVATION_BUFFER_SIZE:]
        if len(self.OBSERVATION_KILL_BUFFER) == self.OBSERVATION_BUFFER_SIZE and self.mutant_number % self.OBSERVATION_UPDATE_DELTA == 0:
            self.loss_o = training_model(self.observation_nn, [observation_to_tensor(x[0][0], x[0][1], self.num_actions) for x in self.OBSERVATION_KILL_BUFFER], [torch.FloatTensor([x[1]]) for x in self.OBSERVATION_KILL_BUFFER], self.observation_nn_opt, self.observation_loss_function)

        state = self.State(env.test_sequence, mutant_operators_list.index(self.mutant["operator"]), action, env.num_of_mutants_same_operator_killed)

        terminal_state = False
        if len(env.test_sequence) == len(self.tests) or killed:
            terminal_state = True

        return state, killed, terminal_state


    def run(self, mutant, mutant_number, mutant_not_killable):

        self.mutant_number = mutant_number

        self.mutant = mutant

        self.mutant_not_killable = mutant_not_killable

        self.done = False

        if self.tree:
            # This is not the first run, we select the root node based on the UCT formula
            scores = [(child.T / child.N) + self.c * sqrt(log(self.mutant_number) / child.N) if child.N > 0 else float('inf') for child in self.root_nodes]
            possible_trees_indexes = [i for i, x in enumerate(scores) if x == max(scores)]
            self.tree = self.root_nodes[random.choice(possible_trees_indexes)]
            self.tree.done = False
            initial_state = self.State([self.tree.action_index], mutant_operators_list.index(self.mutant["operator"]),
                                         self.tree.action_index, self.kills_matrix[self.mutant["operator"]][self.tests[self.tree.action_index]["test_id"]])
            # since we are at the root node, we immediately execute the action and get the observation
            _, killed, terminal_state = self.take_step(self.tree.action_index, initial_state)
            self.tree.observation = initial_state
            self.tree.killed = killed
        else:
            # This is the first run, we create the root nodes for each action and select one at random
            self.init_tree()
            node_index = random.choice(range(len(self.tests)))
            self.tree = self.root_nodes[node_index]
            _, killed, terminal_state = self.take_step(node_index, self.tree.observation)
            self.tree.killed = killed
            self.tree.done = terminal_state

            if self.tree.done:
                done = True
                self.tree.T += len(self.tests) / len(self.tests)

        loss_v = None
        loss_p = None
        reward_e = 0

        obs = []
        ps = []
        p_obs = []

        step = 1

        tree = self.tree

        while not self.done:

            if self.tree.parent is None and self.tree.killed:
                # Root node killed, we can stop the episode
                reward_e = len(self.tests) - step
                self.tree.T += len(self.tests) / len(self.tests)
                self.done = True
                break

            step = step + 1

            mytree, action, ob, p, p_ob = self.policy_player_mcts(tree)

            actual_observation, killed, terminal_state = self.take_step(action, ob)

            # We update the tree with the actual observation after executing the action, thereby replacing the placeholder
            mytree.observation = actual_observation
            mytree.killed = killed
            mytree.done = terminal_state
            ob = actual_observation

            tree = mytree

            self.done = mytree.done

            obs.append(ob)
            ps.append(p)
            p_obs.append(p_ob)

            current_reward = len(self.tests) - step

            print("Step: ", step, "Sequence: ", ob.test_sequence, "Reward: ", current_reward)

            if self.done:
                mytree.T += current_reward / len(self.tests) # update the value of the node with the actual reward
                reward_e = current_reward
                self.replay_buffer.add(obs[-1], current_reward, ps[-1], p_obs[-1])
                break

        print('Episode reward: ' + str(reward_e))

        # Networks update

        if (self.mutant_number + 1) % self.UPDATE_DELTA == 0 and len(self.replay_buffer) > self.BATCH_SIZE:

            experiences = self.replay_buffer.sample()

            # Each state has as target value the rewards of the episode

            inputs = [observation_to_tensor(experience.obs, total_number_of_tests=self.num_actions) for experience in experiences]
            targets = [torch.FloatTensor([experience.v / (self.max_reward)]) for experience in experiences]

            k = 1.0 # parameter to balance the training set, we keep only k% of the inputs and targets with rewards == 0

            balanced_inputs = []
            balanced_targets = []

            # keep only k% of inputs and targets with rewards == 0
            for i in range (len(targets)):
                if experiences[i].v != 0 or random.random() < k:
                    balanced_inputs.append(inputs[i])
                    balanced_targets.append(targets[i])

            inputs = balanced_inputs
            targets = balanced_targets

            loss_v = training_model(self.value_net, inputs, targets, self.value_opt, self.value_loss_function)

            # Each state has as target policy the policy from the next state function

            inputs = [observation_to_tensor(experience.p_obs, total_number_of_tests=self.num_actions) for experience in experiences]
            targets = [torch.FloatTensor(experience.p) for experience in experiences if experience.p is not None]

            balanced_inputs = []
            balanced_targets = []

            for i in range (len(targets)):
                if experiences[i].v != 0 or random.random() < k:
                    balanced_inputs.append(inputs[i])
                    balanced_targets.append(targets[i])

            inputs = balanced_inputs
            targets = balanced_targets

            loss_p = training_model(self.policy_net, inputs, targets, self.policy_opt, self.policy_loss_function)

        return (reward_e, loss_v, loss_p, self.loss_o, self.number_of_tests_executed, self.number_of_tests_executed_on_killable_mutants,
                self.UPDATE_DELTA, self.current_sut_tests_execution_time)
