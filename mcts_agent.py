import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import time

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


class InferenceType(Enum):
    VALUE = "value"
    POLICY = "policy"


@dataclass
class InferenceRequest:
    mutant_id: int
    inference_type: InferenceType
    observation: Any
    action: Optional[int] = None
    node_id: Optional[int] = None


@dataclass
class SearchContext:
    mutant_id: int
    mutant: Dict
    mutant_not_killable: bool
    current_node: Any
    step: int
    obs_history: List
    ps_history: List
    p_obs_history: List
    reward_e: float
    done: bool
    waiting_for_inference: bool
    pending_inference_type: Optional[InferenceType] = None
    pending_action: Optional[int] = None
    pending_node_id: Optional[int] = None  # Aggiungi questo campo


class InferenceManager:

    def __init__(self, mcts_agent, inference_batch_size, window_size):
        self.mcts_agent = mcts_agent
        self.inference_batch_size = inference_batch_size
        self.window_size = window_size

        self.pending_requests = []
        self.requests_lock = threading.Lock()

        self.result_dict = {}
        self.result_lock = threading.Lock()

        self.running = True
        self.trigger_event = threading.Event()
        self.mutants_processed_count = 0
        self.count_lock = threading.Lock()

        self.max_wait_time = 0.5

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("[InferenceManager] Thread started")

    def request_inference(self, request: InferenceRequest):
        with self.requests_lock:
            self.pending_requests.append(request)

    def increment_mutant_count(self):
        with self.count_lock:
            self.mutants_processed_count += 1
            if self.mutants_processed_count >= self.window_size:
                self.mutants_processed_count = 0
                self.trigger_event.set()

    def get_result(self, mutant_id: int, inference_type: InferenceType):
        with self.result_lock:
            if mutant_id in self.result_dict and inference_type in self.result_dict[mutant_id]:
                result = self.result_dict[mutant_id].pop(inference_type)
                if not self.result_dict[mutant_id]:
                    del self.result_dict[mutant_id]
                return result
        return None

    def has_result(self, mutant_id: int, inference_type: InferenceType) -> bool:
        with self.result_lock:
            return mutant_id in self.result_dict and inference_type in self.result_dict[mutant_id]

    def _worker(self):
        while self.running:
            # Wait for trigger or timeout
            triggered = self.trigger_event.wait(timeout=0.1)

            if not self.running:
                break

            self.trigger_event.clear()

            # Get all pending requests
            with self.requests_lock:
                if not self.pending_requests:
                    continue

                batch = self.pending_requests.copy()
                self.pending_requests.clear()

            print(f"[InferenceManager] Processing {len(batch)} requests")
            self._process_batch(batch)

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process batch of requests."""
        # Group by type
        by_type = {InferenceType.VALUE: [], InferenceType.POLICY: []}
        for req in batch:
            by_type[req.inference_type].append(req)

        # Process each type
        for inf_type, requests in by_type.items():
            if not requests:
                continue

            # Process in chunks
            for i in range(0, len(requests), self.inference_batch_size):
                chunk = requests[i:i + self.inference_batch_size]

                # Prepare inputs
                if inf_type == InferenceType.VALUE:
                    model = self.mcts_agent.value_net
                    inputs = [observation_to_tensor(req.observation, total_number_of_tests=self.mcts_agent.num_actions)
                             for req in chunk]
                elif inf_type == InferenceType.POLICY:
                    model = self.mcts_agent.policy_net
                    inputs = [observation_to_tensor(req.observation, total_number_of_tests=self.mcts_agent.num_actions)
                             for req in chunk]

                # Batch inference
                inputs_tensor = torch.stack(inputs).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    outputs = model(inputs_tensor)

                # Store results
                with self.result_lock:
                    for j, req in enumerate(chunk):
                        if req.mutant_id not in self.result_dict:
                            self.result_dict[req.mutant_id] = {}

                        if inf_type == InferenceType.VALUE:
                            result = outputs[j].item()
                        else:
                            result = torch.softmax(outputs[j], dim=-1).cpu().numpy()

                        self.result_dict[req.mutant_id][inf_type] = result

    def shutdown(self):
        self.running = False
        self.trigger_event.set()
        self.worker_thread.join(timeout=2.0)


class TrainingManager:

    def __init__(self, mcts_agent):
        self.mcts_agent = mcts_agent
        self.training_queue = queue.Queue()
        self.latest_losses = {'value': None, 'policy': None}
        self.losses_lock = threading.Lock()
        self.running = True

        # Start training thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("[TrainingManager] Thread started")

    def request_training(self):
        try:
            self.training_queue.put_nowait(True)
        except queue.Full:
            pass  # Skip if queue is full

    def _worker(self):
        while self.running:
            try:
                self.training_queue.get(timeout=0.5)

                if len(self.mcts_agent.replay_buffer) >= self.mcts_agent.BATCH_SIZE:
                    experiences = self.mcts_agent.replay_buffer.sample()

                    # Train value network
                    inputs = [observation_to_tensor(exp.obs, total_number_of_tests=self.mcts_agent.num_actions)
                             for exp in experiences]
                    targets = [torch.FloatTensor([exp.v / self.mcts_agent.max_reward]) for exp in experiences]

                    balanced_inputs = []
                    balanced_targets = []
                    k = 1.0

                    for i in range(len(targets)):
                        if experiences[i].v != 0 or random.random() < k:
                            balanced_inputs.append(inputs[i])
                            balanced_targets.append(targets[i])

                    loss_v = None
                    if balanced_inputs:
                        loss_v = training_model(self.mcts_agent.value_net, balanced_inputs, balanced_targets,
                                               self.mcts_agent.value_opt, self.mcts_agent.value_loss_function)

                    # Train policy network
                    inputs = [observation_to_tensor(exp.p_obs, total_number_of_tests=self.mcts_agent.num_actions)
                             for exp in experiences if exp.p is not None]
                    targets = [torch.FloatTensor(exp.p) for exp in experiences if exp.p is not None]

                    balanced_inputs = []
                    balanced_targets = []

                    for i in range(min(len(inputs), len(targets))):
                        if i < len(experiences) and (experiences[i].v != 0 or random.random() < k):
                            balanced_inputs.append(inputs[i])
                            balanced_targets.append(targets[i])

                    loss_p = None
                    if balanced_inputs:
                        loss_p = training_model(self.mcts_agent.policy_net, balanced_inputs, balanced_targets,
                                               self.mcts_agent.policy_opt, self.mcts_agent.policy_loss_function)

                    with self.losses_lock:
                        self.latest_losses['value'] = loss_v
                        self.latest_losses['policy'] = loss_p

            except queue.Empty:
                continue

    def get_latest_losses(self):
        with self.losses_lock:
            return self.latest_losses['value'], self.latest_losses['policy']

    def shutdown(self):
        self.running = False
        self.worker_thread.join(timeout=2.0)


class AsymmetricLoss(nn.Module):
    def __init__(self, alpha):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        errors = targets - predictions
        loss = torch.where(errors < 0, self.alpha * errors ** 2, errors ** 2)
        return loss.mean()


class MCTSAgent:

    def __init__(self, policy_nn=None, value_nn=None, tests=None, kills_matrix=None,
                 sut_name=None, number_of_mutants=None, buffer_size=None, batch_size=None, update_delta=None,
                 rollout_after=None, asymmetric_loss_alpha=None, c_parameter=None, value_network_learning_rate=None,
                 policy_network_learning_rate=None, inference_batch_size=128, window_size=32):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using device: ", device)
            print(torch.cuda.get_device_name(0))

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
        self.max_reward = len(tests)
        self.kills_ranking = {test: 0 for test in range(len(self.tests))}
        self.current_sut_tests_execution_time = 0

        # Mutant-related
        self.mutant_count = 0
        self.number_of_tests_executed = 0
        self.number_of_tests_executed_on_killable_mutants = 0

        # MCTS stuff
        self.c = c_parameter
        self.tree = None
        self.root_nodes = []
        self.node_counter = 0

        # Parallelization parameters
        self.window_size = window_size
        self.inference_batch_size = inference_batch_size

        # Inference and training management (SEPARATE THREADS)
        self.inference_manager = InferenceManager(self, inference_batch_size, window_size)
        self.training_manager = TrainingManager(self)

    def _get_node_id(self):
        """Generate unique node ID."""
        self.node_counter += 1
        return self.node_counter

    def init_tree(self):
        for i in range(len(self.tests)):
            initial_state = self.State([i], mutant_operators_list.index(self.mutant["operator"]), i)
            node = self.Node(False, False, None, initial_state, i, self, self._get_node_id())
            self.root_nodes.append(node)

    class State:
        def __init__(self, test_sequence: list, mutant_operator: int, test_index: int = 0):
            self.test_sequence = test_sequence
            self.mutant_operator = mutant_operator
            self.test_index = test_index

    class Node:
        def __init__(self, done, killed, parent, observation, action_index, mcts_agent, node_id):
            self.mcts_agent = mcts_agent
            self.children = {}
            self.T = 0
            self.N = 0
            self.observation = observation
            self.done = done
            self.killed = killed
            self.parent = parent
            self.backup_parent = parent
            self.action_index = action_index
            self.nn_v = 0
            self.nn_p = [0] * self.mcts_agent.num_actions
            self.node_id = node_id

        def getUCBscore(self):
            if self.N == 0:
                return float('inf')

            top_node = self
            if top_node.parent:
                top_node = top_node.parent

            value_score = (self.T / self.N)
            prior_score = 0
            if self.mcts_agent.mutant_count >= self.mcts_agent.ROLLOUT_AFTER:
                prior_score = self.mcts_agent.c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)

            return value_score + prior_score

        def detach_parent(self):
            del self.parent
            self.parent = None

        def get_available_actions(self):
            return [i for i in range(len(self.mcts_agent.tests)) if i not in self.observation.test_sequence]

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

            next_child = list(self.children.items())[np.argmax(probs)][1]

            masked_probs = [0] * self.mcts_agent.num_actions
            for index, child in enumerate(self.children.values()):
                masked_probs[child.action_index] = probs[index]

            next_child.observation = self.mcts_agent.State(
                next_child.observation.test_sequence,
                mutant_operators_list.index(self.mcts_agent.mutant["operator"]),
                next_child.action_index
            )

            return next_child, next_child.action_index, next_child.observation, masked_probs, self.observation

    def select_action(self, node, mutant_id: int) -> Tuple[Optional[int], bool]:
        """
        Select action for expansion. Returns (action, needs_rollout).
        If needs_rollout is True, rollout was requested and we should wait.
        """
        possible_actions = node.get_available_actions()

        if len(possible_actions) == 0:
            return None, False

        # Check if we need rollout for this node
        if self.mutant_count >= self.ROLLOUT_AFTER:
            if node.nn_v == 0 and all(p == 0 for p in node.nn_p):
                # Request rollout
                self.inference_manager.request_inference(
                    InferenceRequest(mutant_id, InferenceType.VALUE, node.observation, node_id=node.node_id)
                )
                self.inference_manager.request_inference(
                    InferenceRequest(mutant_id, InferenceType.POLICY, node.observation, node_id=node.node_id)
                )
                return None, True

        # Select action based on strategy
        if self.mutant_count < self.ROLLOUT_AFTER:
            action = None
            for i in range(len(self.kills_ranking)):
                if list(self.kills_ranking.keys())[i] in possible_actions:
                    action = list(self.kills_ranking.keys())[i]
                    break
            if action is None:
                action = random.choice(possible_actions)
        else:
            action = max(possible_actions, key=lambda x: node.nn_p[x])

        return action, False

    def create_child(self, node, action: int, mutant_id: int) -> 'MCTSAgent.Node':
        env_copy = deepcopy(node.observation)
        env_copy.test_sequence.append(action)

        placeholder_obs = self.State(env_copy.test_sequence, env_copy.mutant_operator, action)
        done = len(env_copy.test_sequence) == len(self.tests)

        child_id = self._get_node_id()
        # Non impostiamo killed qui - verrà impostato dopo l'esecuzione reale del test
        child = self.Node(done, False, node, placeholder_obs, action, self, child_id)
        node.children[action] = child

        return child

    def policy_player_mcts(self, context: SearchContext) -> bool:
        """Execute one step of MCTS. Returns True if waiting for inference."""
        node = context.current_node
        node.N += 1

        if not node.children and not node.done:
            action, needs_rollout = self.select_action(node, context.mutant_id)

            if needs_rollout:
                context.waiting_for_inference = True
                context.pending_inference_type = InferenceType.VALUE
                context.pending_node_id = node.node_id
                return True

            if action is None:
                context.done = True
                return False

            # Crea il figlio e esegui subito il test reale
            child = self.create_child(node, action, context.mutant_id)

            # Esegui il test reale
            actual_observation, killed, terminal_state = self.take_step(action, child.observation)

            # Aggiorna il figlio con i risultati reali
            child.observation = actual_observation
            child.killed = killed
            child.done = terminal_state

            # Ora richiedi rollout se necessario, con lo stato aggiornato
            if self.mutant_count >= self.ROLLOUT_AFTER and not child.done:
                self.inference_manager.request_inference(
                    InferenceRequest(context.mutant_id, InferenceType.VALUE, child.observation, node_id=child.node_id)
                )
                self.inference_manager.request_inference(
                    InferenceRequest(context.mutant_id, InferenceType.POLICY, child.observation, node_id=child.node_id)
                )
                context.waiting_for_inference = True
                context.pending_inference_type = InferenceType.VALUE
                context.pending_action = action
                context.pending_node_id = child.node_id
                return True

            # Se terminato o non serve rollout, aggiorna subito
            context.step += 1
            context.done = child.done

            if context.done:
                reward = len(self.tests) - context.step
                context.reward_e = reward
                child.T += reward / len(self.tests)
                # Aggiungi dummy p per compatibilità
                dummy_p = [1.0 / len(self.tests)] * len(self.tests)
                self.replay_buffer.add(actual_observation, reward, dummy_p, node.observation)

            return False

        if node.children:
            try:
                next_tree, next_action, obs, p, p_obs = node.next()
                next_tree.detach_parent()

                context.current_node = next_tree
                context.obs_history.append(obs)
                context.ps_history.append(p)
                context.p_obs_history.append(p_obs)

                # Esegui il test reale
                actual_observation, killed, terminal_state = self.take_step(next_action, obs)

                # Aggiorna con i risultati reali
                next_tree.observation = actual_observation
                next_tree.killed = killed
                next_tree.done = terminal_state

                context.step += 1
                context.done = terminal_state

                if context.done:
                    reward = len(self.tests) - context.step
                    context.reward_e = reward
                    next_tree.T += reward / len(self.tests)
                    self.replay_buffer.add(actual_observation, reward, p, p_obs)

                return False

            except ValueError:
                context.done = True
                return False

        context.done = True
        return False

    def resume_context(self, context: SearchContext) -> bool:
        """Try to resume context. Returns True if still waiting."""
        if context.pending_inference_type == InferenceType.VALUE:
            v_result = self.inference_manager.get_result(context.mutant_id, InferenceType.VALUE)
            p_result = self.inference_manager.get_result(context.mutant_id, InferenceType.POLICY)

            if v_result is None or p_result is None:
                # Inizializza il contatore di attesa se non esiste
                if not hasattr(context, 'inference_wait_started'):
                    context.inference_wait_started = time.time()

                elapsed_time = time.time() - context.inference_wait_started

                # Se abbiamo aspettato troppo, forza il trigger del batch
                if elapsed_time > self.inference_manager.max_wait_time:
                    print(f"[MCTS] Warning: Long wait for mutant {context.mutant_id} ({elapsed_time:.1f}s), forcing batch processing")
                    self.inference_manager.trigger_event.set()
                    # Dai più tempo al thread di inferenza
                    time.sleep(0.1)

                    # Controlla di nuovo
                    v_result = self.inference_manager.get_result(context.mutant_id, InferenceType.VALUE)
                    p_result = self.inference_manager.get_result(context.mutant_id, InferenceType.POLICY)

                    if v_result is None or p_result is None:
                        # Se ancora non abbiamo risultati, aspetta ancora un po'
                        return True

                return True

            # Reset timer on success
            if hasattr(context, 'inference_wait_started'):
                delattr(context, 'inference_wait_started')

            node = context.current_node
            if not node.children:
                node.nn_v = v_result if v_result > 0 else 0
                node.nn_p = p_result
                node.T += node.nn_v
            else:
                child = node.children.get(context.pending_action)
                if child:
                    child.nn_v = v_result if v_result > 0 else 0
                    child.nn_p = p_result
                    child.T += child.nn_v

            context.waiting_for_inference = False
            return False

        return False

    def execute_test_on_mutant(self, test, mutant):
        if not self.mutant_not_killable:
            self.number_of_tests_executed_on_killable_mutants += 1

        self.number_of_tests_executed += 1

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

        if outcome == 1:
            self.kills_ranking[self.tests.index(test)] = self.kills_ranking[self.tests.index(test)] + 1 if self.tests.index(test) in self.kills_ranking.keys() else 1
            self.kills_ranking = dict(sorted(self.kills_ranking.items(), key=lambda item: item[1], reverse=True))

        return outcome

    def take_step(self, action, env):
        """Perform all necessary actions on the environment after selecting an action."""
        killed = self.execute_test_on_mutant(self.tests[action], self.mutant)

        state = self.State(env.test_sequence, env.mutant_operator, action)
        terminal_state = len(env.test_sequence) == len(self.tests) or killed

        return state, killed, terminal_state

    def run(self, mutants):
        """
        Process all mutants with parallel inference and training.
        MCTS runs continuously, inference and training in separate threads.
        """
        print(f"[MCTS] Starting with {len(mutants)} mutants, window={self.window_size}, batch={self.inference_batch_size}")

        # Initialize contexts
        contexts = []
        for idx, mutant in enumerate(mutants):
            no_test_killing = True
            if len(mutant["testResults"]) > 0:
                for test_file in mutant["testResults"]:
                    if len(mutant["testResults"][test_file]["failed"]) > 0:
                        no_test_killing = False
                        break

            context = SearchContext(
                mutant_id=idx,
                mutant=mutant,
                mutant_not_killable=no_test_killing,
                current_node=None,
                step=1,
                obs_history=[],
                ps_history=[],
                p_obs_history=[],
                reward_e=0,
                done=False,
                waiting_for_inference=False
            )
            contexts.append(context)

        # MAIN LOOP - Process mutants in order
        current_idx = 0
        total_done = 0
        mutants_processed_since_batch = 0
        consecutive_waiting_loops = 0
        last_progress_time = time.time()

        while total_done < len(contexts):
            context = contexts[current_idx]

            # Set global state
            self.mutant_count = context.mutant_id
            self.mutant = context.mutant
            self.mutant_not_killable = context.mutant_not_killable

            # Initialize if needed
            if not context.done and context.current_node is None:
                if self.tree:
                    scores = [(child.T / child.N) + self.c * sqrt(log(self.mutant_count + 1) / child.N)
                             if child.N > 0 else float('inf') for child in self.root_nodes]
                    possible_trees_indexes = [i for i, x in enumerate(scores) if x == max(scores)]
                    tree = self.root_nodes[random.choice(possible_trees_indexes)]
                    tree.done = False
                    initial_state = self.State([tree.action_index],
                                               mutant_operators_list.index(context.mutant["operator"]),
                                               tree.action_index)
                    _, killed, terminal_state = self.take_step(tree.action_index, initial_state)
                    tree.observation = initial_state
                    tree.killed = killed
                    tree.done = terminal_state or killed
                    context.current_node = tree
                else:
                    self.init_tree()
                    node_index = random.choice(range(len(self.tests)))
                    tree = self.root_nodes[node_index]
                    _, killed, terminal_state = self.take_step(node_index, tree.observation)
                    tree.killed = killed
                    tree.done = terminal_state or killed
                    if tree.done:
                        tree.T += len(self.tests) / len(self.tests)
                    context.current_node = tree

                if context.current_node.parent is None and context.current_node.killed:
                    context.reward_e = len(self.tests) - context.step + 1
                    context.current_node.T += len(self.tests) / len(self.tests)
                    context.done = True
                    total_done += 1

            # Try to advance if not waiting
            made_progress = False
            if not context.done:
                if context.waiting_for_inference:
                    # Try to resume
                    still_waiting = self.resume_context(context)
                    if not still_waiting:
                        made_progress = True
                        # Advance after resume
                        while not context.done and not context.waiting_for_inference:
                            self.policy_player_mcts(context)
                            if context.waiting_for_inference or context.done:
                                break

                        if context.done:
                            total_done += 1
                            # Trigger training
                            if total_done % self.UPDATE_DELTA == 0:
                                self.training_manager.request_training()
                else:
                    made_progress = True
                    # Advance freely
                    while not context.done and not context.waiting_for_inference:
                        self.policy_player_mcts(context)
                        if context.waiting_for_inference or context.done:
                            break

                    if context.done:
                        total_done += 1
                        # Trigger training
                        if total_done % self.UPDATE_DELTA == 0:
                            self.training_manager.request_training()

            # Move to next mutant
            prev_idx = current_idx
            current_idx = (current_idx + 1) % len(contexts)

            # Count mutants processed for window trigger
            if made_progress:
                mutants_processed_since_batch += 1
                consecutive_waiting_loops = 0
            else:
                consecutive_waiting_loops += 1

            # Trigger inference batch after window_size mutants processed
            if mutants_processed_since_batch >= self.window_size:
                print(f"[MCTS] Triggering batch inference (processed {mutants_processed_since_batch} mutants)")
                self.inference_manager.trigger_event.set()
                mutants_processed_since_batch = 0
                time.sleep(0.05)

            # If we've looped through all mutants without progress, wait for inference
            if consecutive_waiting_loops >= len(contexts):
                active_mutants = sum(1 for ctx in contexts if not ctx.done)
                waiting_mutants = sum(1 for ctx in contexts if ctx.waiting_for_inference and not ctx.done)

                if waiting_mutants > 0 and waiting_mutants == active_mutants:
                    print(f"[MCTS] All {waiting_mutants} active mutants waiting for inference, triggering batch...")
                    self.inference_manager.trigger_event.set()
                    time.sleep(0.2)
                    consecutive_waiting_loops = 0

            # Forza il batch se ci sono richieste in attesa da troppo tempo
            current_time = time.time()
            if current_time - last_progress_time > self.inference_manager.max_wait_time:
                with self.inference_manager.requests_lock:
                    if len(self.inference_manager.pending_requests) > 0:
                        print(f"[MCTS] Forcing batch processing ({len(self.inference_manager.pending_requests)} pending requests)")
                        self.inference_manager.trigger_event.set()
                        time.sleep(0.1)
                last_progress_time = current_time

            # Progress logging
            if total_done > 0 and total_done % 10 == 0:
                if prev_idx == 0 and current_idx != 0:
                    waiting = sum(1 for ctx in contexts if ctx.waiting_for_inference and not ctx.done)
                    active = sum(1 for ctx in contexts if not ctx.done and not ctx.waiting_for_inference)
                    print(f"[MCTS] Progress: {total_done}/{len(contexts)} done, {waiting} waiting, {active} active")

        # Final training
        self.training_manager.request_training()
        time.sleep(1.0)  # Wait for final training

        # Get results
        avg_reward = sum(ctx.reward_e for ctx in contexts) / len(contexts) if contexts else 0
        loss_v, loss_p = self.training_manager.get_latest_losses()

        # Shutdown threads
        self.inference_manager.shutdown()
        self.training_manager.shutdown()

        print(f"[MCTS] Completed. Avg reward: {avg_reward:.2f}")

        return (avg_reward, loss_v, loss_p, self.number_of_tests_executed, self.number_of_tests_executed_on_killable_mutants,
                self.UPDATE_DELTA, self.current_sut_tests_execution_time)
