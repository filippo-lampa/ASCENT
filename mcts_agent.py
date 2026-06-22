import numpy as np

from copy import deepcopy
from math import *
import random
import logging

import torch
from sympy.physics.units import current
from torch import nn

from replay_buffer import ReplayBuffer
from networks.utility import inference, training_model, observation_to_tensor
from utils.consts import mutant_operators_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device: ", device)


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

    def __init__(self, policy_nn=None, value_nn=None, tests=None, kills_matrix=None,
                 sut_name=None, number_of_mutants=None, buffer_size=None, batch_size=None, update_delta=None,
                 rollout_after=None, asymmetric_loss_alpha=None, c_parameter=None, value_network_learning_rate=None,
                 policy_network_learning_rate=None, agents_manager=None, agent_key=None,
                 diversity_bonus_weight=0.0):

        # Init Neural-MCTS parameters
        self.ROLLOUT_AFTER = rollout_after
        self.BUFFER_SIZE = number_of_mutants if buffer_size is None else buffer_size
        self.BATCH_SIZE = batch_size
        self.UPDATE_DELTA = update_delta
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE)

        # Init networks
        self.policy_net = policy_nn.to(device)
        self.value_net = value_nn.to(device)
        self.asymmetric_loss_alpha = asymmetric_loss_alpha
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=value_network_learning_rate)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=policy_network_learning_rate)
        self.value_loss_function = AsymmetricLoss(self.asymmetric_loss_alpha)
        self.policy_loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.5)

        # Others
        self.tests = tests
        self.sut_name = sut_name
        self.num_actions = len(self.tests)
        self.max_reward = len(tests)  # the maximum reward of the current episode to scale the values
        self.kills_ranking = {test: 0 for test in range(
            len(self.tests))}  # kills ranking of the tests, used as heuristic until we start relying on the networks
        self.done = False  # Checks if the episode is done (the mutant is killed or we run out of tests)
        self.current_sut_tests_execution_time = 0
        self.agents_manager = agents_manager

        self.agent_key = agent_key #"exploration_proposed_test", "exploitation_proposed_test", "diversity_proposed_test"

        self.diversity_bonus_weight = diversity_bonus_weight # When > 0, the UCB score gains an extra term that boosts tests which have been executed less often globally. Only used by the diversity agent.

        # Per-episode bookkeeping (reset at the start of every mutant's episode)
        self.obs = []
        self.ps = []
        self.p_obs = []
        self.step = 0
        self.reward_e = 0

        # Mutant-related stuff
        self.mutant_number = None
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
            initial_state = self.State([i], mutant_operators_list.index(self.mutant["operator"]), i)
            self.tree = self.Node(False, False, None, initial_state, i, self)
            self.root_nodes.append(self.tree)

    class State:
        def __init__(self, test_sequence: list[int], mutant_operator: str, test_index: int = 0):
            self.test_sequence = test_sequence
            self.mutant_operator = mutant_operator
            self.test_index = test_index

    class Node:
        def __init__(self, done, killed, parent, observation, action_index, mcts_agent):
            self.mcts_agent = mcts_agent
            self.children = {}
            self.T = 0  # sum of rewards
            self.N = 0  # number of visits
            self.observation = observation  # state of the prioritization
            self.done = done  # if the node is terminal. could be because we ran out of tests or because the mutant is killed
            self.killed = killed  # if the mutant is killed
            self.parent = parent
            self.backup_parent = parent
            self.action_index = action_index  # action index that leads to this node
            self.nn_v = 0  # value from the value network
            self.nn_p = [0] * self.mcts_agent.num_actions  # priors from the policy network

        def getUCBscore(self):
            top_node = self
            if top_node.parent:
                top_node = top_node.parent

            exploration = sqrt(top_node.N) / (1 + self.N)

            # exploitation
            value_score = self.T / (1 + self.N) if self.N > 0 else 0

            # exploration
            prior_score = 0
            if self.mcts_agent.mutant_number >= self.mcts_agent.ROLLOUT_AFTER:
                prior_probability = self.parent.nn_p[self.action_index] if self.parent else (
                            1.0 / self.mcts_agent.num_actions)
                prior_score = self.mcts_agent.c * prior_probability * exploration
            else:
                uniform_prior = 1.0 / self.mcts_agent.num_actions
                prior_score = self.mcts_agent.c * uniform_prior * exploration

            diversity_score = 0
            if self.mcts_agent.diversity_bonus_weight > 0:
                exec_count = self.mcts_agent.agents_manager.test_execution_counts[self.action_index]
                diversity_score = self.mcts_agent.diversity_bonus_weight / (1 + exec_count)

            return value_score + prior_score + diversity_score

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
                logging.debug("no possible actions")
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
                # choose the action index that is in the possible actions and has the highest policy score
                action = max(possible_actions, key=lambda x: self.nn_p[x])

            env_copy = deepcopy(self.observation)
            env_copy.test_sequence.append(action)

            placeholder_observation = self.mcts_agent.State(env_copy.test_sequence, env_copy.mutant_operator, action)
            killed = False
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

        def get_ranking(self):
            if self.done:
                raise ValueError("episode has ended")

            if not self.children:
                raise ValueError('no children found and episode hasn\'t ended')

            for child in self.children.values():
                child.parent = child.backup_parent

            # Safely extract PUCT scores without fear of infinity values
            scores = [c.getUCBscore() for c in self.children.values()]
            max_U = max(scores)

            if max_U == 0:
                probs = [1 / len(self.children) for _ in self.children.values()]
            else:
                probs = [float(score / max_U) for score in scores]

            probs = np.array(probs)
            probs = probs / probs.sum()

            # mask probabilities of all other actions to 0
            masked_probs = [0] * self.mcts_agent.num_actions
            for index, child in enumerate(self.children.values()):
                masked_probs[child.action_index] = probs[index]

            self.nn_p = masked_probs

            return masked_probs

        def select_child(self, chosen_action):
            '''
            Given the test index chosen by the AgentsManager (after aggregating the
            rankings of all the agents), return the corresponding child node, fitted
            to the current mutant. If this node doesn't have a child for `chosen_action` yet (e.g. because
            another agent's ranking led the aggregator towards an action this agent
            hadn't expanded), the child is created on the fly.
            '''

            if chosen_action not in self.children:
                env_copy = deepcopy(self.observation)
                env_copy.test_sequence.append(chosen_action)

                placeholder_observation = self.mcts_agent.State(env_copy.test_sequence, env_copy.mutant_operator,
                                                                chosen_action)
                done = True if len(env_copy.test_sequence) == len(self.mcts_agent.tests) else False

                self.children[chosen_action] = type(self)(done, False, self, placeholder_observation, chosen_action,
                                                          self.mcts_agent)
                self.children[chosen_action].nn_v, self.children[chosen_action].nn_p = self.children[
                    chosen_action].rollout()

                if self.mcts_agent.mutant_number >= self.mcts_agent.ROLLOUT_AFTER:
                    self.children[chosen_action].T += self.children[chosen_action].nn_v

            next_child = self.children[chosen_action]

            # fit the chosen node to the current mutant
            next_child.observation = self.mcts_agent.State(next_child.observation.test_sequence,
                                                           mutant_operators_list.index(
                                                               self.mcts_agent.mutant["operator"]),
                                                           next_child.action_index)

            return next_child, next_child.action_index, next_child.observation, self.nn_p, self.observation

    def _update_kills_ranking(self, action_index):
        '''
        Update the heuristic kills_ranking with the test that just killed the mutant.
        '''
        if action_index in self.kills_ranking:
            self.kills_ranking[action_index] += 1
        else:
            self.kills_ranking[action_index] = 1

        self.kills_ranking = dict(sorted(self.kills_ranking.items(), key=lambda item: item[1], reverse=True))

    def prepare_episode(self, mutant, mutant_number, mutant_not_killable):
        '''
        Reset the per-episode state of this agent before prioritizing a new mutant.
        Must be called (for every agent) before propose_root_ranking().
        '''

        self.mutant_number = mutant_number
        self.mutant = mutant
        self.mutant_not_killable = mutant_not_killable

        self.done = False
        self.obs = []
        self.ps = []
        self.p_obs = []
        self.step = 1  # the root selection counts as the first step of the episode
        self.reward_e = 0

        if self.tree is None:
            # This is the first ever episode for this agent: create the root nodes for each action
            self.init_tree()

    def propose_root_ranking(self):
        '''
        Compute this agent's ranking for the first test using PUCT logic,
        allowing agents with different 'c' parameters to disagree immediately.
        '''
        scores = []

        # Total root level focus can be scaled by the current mutant number
        parent_n = max(1, self.mutant_number)

        for child in self.root_nodes:
            # Value score fallback if never visited
            value_score = (child.T / child.N) if child.N > 0 else 0.0

            # Smooth denominator exploration component
            exploration = sqrt(parent_n) / (1 + child.N)

            prior_score = 0
            if self.mutant_number >= self.ROLLOUT_AFTER:
                # If networks are active, pull from your initialized node policy vector or a default prior
                prior_prob = child.nn_p[child.action_index] if hasattr(child, 'nn_p') and child.nn_p else (
                            1.0 / self.num_actions)
                prior_score = self.c * prior_prob * exploration
            else:
                # Cold-start uniform prior so that different 'c' values cause early disagreement
                uniform_prior = 1.0 / self.num_actions
                prior_score = self.c * uniform_prior * exploration

            scores.append(value_score + prior_score)

        max_score = max(scores)

        if max_score == 0:
            probs = [1 / len(scores) for _ in scores]
        else:
            probs = [s / max_score for s in scores]

        probs = np.array(probs)
        probs = probs / probs.sum()

        ranking = [0.0] * self.num_actions
        for index, child in enumerate(self.root_nodes):
            ranking[child.action_index] = probs[index]

        return ranking

    def apply_root(self, chosen_action, killed, terminal_state):
        '''
        Apply the test chosen by the AgentsManager (after aggregating the agents'
        root rankings) as the first executed test of the episode, and update this
        agent's tree accordingly.
        '''

        self.tree = self.root_nodes[chosen_action]
        self.tree.done = False
        self.tree.observation = self.State([chosen_action], mutant_operators_list.index(self.mutant["operator"]),
                                           chosen_action)
        self.tree.killed = killed
        self.tree.done = terminal_state

        if killed == 1:
            self._update_kills_ranking(chosen_action)

        self.done = terminal_state

        if self.done:
            current_reward = len(self.tests) - self.step
            self.reward_e = current_reward
            self.tree.T += current_reward / len(self.tests)

    def propose_step_ranking(self):
        '''
        Perform one MCTS iteration on the current node of this agent's tree
        (rollout and child expansion) and compute the resulting ranking (score
        distribution over all tests), based on the UCB scores of the node's
        children. The ranking is then shared with the AgentsManager, which
        aggregates it together with the other agents' rankings to decide which
        test is actually executed next.
        '''

        self.step += 1

        mytree = self.tree
        mytree.N += 1  # We increment the number of visits of the current node

        if self.mutant_number >= self.ROLLOUT_AFTER:
            # perform a rollout to fit the current node to the current mutant, so that the next ucb score will be calculated
            # with the current mutant in mind
            nn_v, nn_p = mytree.rollout()
            mytree.nn_v = nn_v
            mytree.nn_p = nn_p
            mytree.T += nn_v

        mytree.create_child()

        return mytree.get_ranking()

    def apply_step(self, chosen_action, killed, terminal_state, test_sequence):
        '''
        Apply the test chosen by the AgentsManager (after aggregating the agents'
        rankings) as the next executed test of the episode, navigating this
        agent's tree to (or creating, if needed) the corresponding child node.

        `test_sequence` is the sequence of test indices executed so far in the
        episode (shared across all agents, since the chosen test is the same for
        all of them); it is used to fit the observation of the resulting node.
        '''

        mytree = self.tree

        next_tree, next_action, _, p, p_obs = mytree.select_child(chosen_action)

        # we detach the current node, keeping only the sub-tree rooted at the chosen action
        next_tree.detach_parent()

        actual_observation = self.State(list(test_sequence), mutant_operators_list.index(self.mutant["operator"]),
                                        next_action)

        next_tree.observation = actual_observation
        next_tree.killed = killed
        next_tree.done = terminal_state

        if killed == 1:
            self._update_kills_ranking(next_action)

        self.tree = next_tree
        self.done = terminal_state

        self.obs.append(actual_observation)
        self.ps.append(p)
        self.p_obs.append(p_obs)

        current_reward = len(self.tests) - self.step

        logging.debug("Agent: ", self.agent_key, "Step: ", self.step, "Sequence: ", actual_observation.test_sequence,
              "Reward: ", current_reward)

        if self.done:
            next_tree.T += current_reward / len(self.tests)  # update the value of the node with the actual reward
            self.reward_e = current_reward
            self.replay_buffer.add(self.obs[-1], current_reward, self.ps[-1], self.p_obs[-1])

    def finish_episode(self):
        '''
        Wrap up the episode for this agent: report the reward obtained and, if it
        is time to update the networks, train them on a batch sampled from the
        replay buffer.
        '''

        logging.debug('Episode reward (' + str(self.agent_key) + '): ' + str(self.reward_e))

        loss_v = None
        loss_p = None

        # Networks update

        if (self.mutant_number + 1) % self.UPDATE_DELTA == 0 and len(self.replay_buffer) > self.BATCH_SIZE:

            experiences = self.replay_buffer.sample()

            # Each state has as target value the rewards of the episode

            inputs = [observation_to_tensor(experience.obs, total_number_of_tests=self.num_actions) for experience in
                      experiences]
            targets = [torch.FloatTensor([experience.v / (self.max_reward)]) for experience in experiences]

            k = 1.0  # parameter to balance the training set, we keep only k% of the inputs and targets with rewards == 0

            balanced_inputs = []
            balanced_targets = []

            # keep only k% of inputs and targets with rewards == 0
            for i in range(len(targets)):
                if experiences[i].v != 0 or random.random() < k:
                    balanced_inputs.append(inputs[i])
                    balanced_targets.append(targets[i])

            inputs = balanced_inputs
            targets = balanced_targets

            loss_v = training_model(self.value_net, inputs, targets, self.value_opt, self.value_loss_function)

            # Each state has as target policy the policy from the next state function

            inputs = [observation_to_tensor(experience.p_obs, total_number_of_tests=self.num_actions) for experience in
                      experiences]
            targets = [torch.FloatTensor(experience.p) for experience in experiences if experience.p is not None]

            balanced_inputs = []
            balanced_targets = []

            for i in range(len(targets)):
                if experiences[i].v != 0 or random.random() < k:
                    balanced_inputs.append(inputs[i])
                    balanced_targets.append(targets[i])

            inputs = balanced_inputs
            targets = balanced_targets

            loss_p = training_model(self.policy_net, inputs, targets, self.policy_opt, self.policy_loss_function)

        return (self.reward_e, loss_v, loss_p)