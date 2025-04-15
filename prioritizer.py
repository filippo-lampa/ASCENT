import argparse
import json
import os
import re
import time

import numpy as np

from mcts_agent import MCTSAgent

from utils.logger import bcolors
from utils.plotting import plot_mutant_prioritization_results
from networks.policy_nn import PolicyNN
from networks.value_nn import ValueNN
from networks.observation_nn import ObservationNN

class Prioritizer:
    '''
    Test prioritizer for mutation testing guided by the MCTS algorithm.
    For each mutant in a set of mutants, the prioritizer will learn to execute the tests in a
    way that maximizes the chance to kill the mutant as soon as possible.
    '''
    def __init__(self, tests_folder_path, mutants_path, sut_name, plot_delta, average_delta, buffer_size, batch_size, update_delta,
                 observation_network_update_delta, observation_network_buffer_size, rollout_after, asymmetric_loss_alpha,
                 c_parameter, value_network_learning_rate, policy_network_learning_rate, observation_network_learning_rate):
        self.tests_folder_path = tests_folder_path
        self.mutants_path = mutants_path
        self.sut_name = sut_name
        self.value_net = None
        self.policy_net = None
        self.observation_net = None
        self.mcts = None
        self.plot_delta = plot_delta
        self.average_delta = average_delta
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_delta = update_delta
        self.observation_network_update_delta = observation_network_update_delta
        self.observation_network_buffer_size = observation_network_buffer_size
        self.rollout_after = rollout_after
        self.asymmetric_loss_alpha = asymmetric_loss_alpha
        self.c_parameter = c_parameter
        self.value_network_learning_rate = value_network_learning_rate
        self.policy_network_learning_rate = policy_network_learning_rate
        self.observation_network_learning_rate = observation_network_learning_rate

    def load_mutants(self):
        '''
        Load mutants from the mutations file.
        '''

        print(f"{bcolors.HEADER}Loading mutants from {self.mutants_path}{bcolors.ENDC}")

        with open(self.mutants_path, 'r', encoding='utf-8') as f:
            mutants = json.load(f)
        mutants_list = []
        for contract, contract_mutants in mutants.items():
            for mutant in contract_mutants:
                if 'operator' not in mutant:
                    print(f"{bcolors.WARNING}Mutant {mutant['id']} has no operator. Skipping...{bcolors.ENDC}")
                    continue
                mutant['contract'] = contract
                mutants_list.append(mutant)

        return mutants_list


    def remove_duplicates(self, tests):
        test_ids = [test['test_id'] for test in tests]
        duplicates = set([test_id for test_id in test_ids if test_ids.count(test_id) > 1])
        if duplicates:
            tests_copy = tests.copy()
            for test in tests_copy:
                if test['test_id'] in duplicates:
                    tests.remove(test)
                    duplicates.remove(test['test_id'])
        return tests

    def load_tests(self):
        """
        load test methods from the tests files within the test folder
        """

        print(f"{bcolors.HEADER}Loading tests from {self.tests_folder_path}{bcolors.ENDC}")

        test_method_names_regex = re.compile(r'it\([\'\"](.*)[\'\"]')
        tests = []
        for root, _, files in os.walk(self.tests_folder_path):
            for test_file in files:
                test_file_path = os.path.join(root, test_file)
                if not os.path.isfile(test_file_path) or not (test_file.endswith('.ts') or test_file.endswith('.js')):
                    continue
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    test_method_names = set(test_method_names_regex.findall(content))
                    tests.extend([
                        {
                            'test_file_path': test_file_path,
                            'test_method_name': test_method_name,
                            'test_file_name': test_file,
                            'test_id': f'{test_file}_{test_method_name}'
                        }
                        for test_method_name in test_method_names
                    ])

        #remove eventual duplicates
        tests = self.remove_duplicates(tests)

        return tests

    def execute(self):
        '''
        Execute the prioritizer.
        '''
        tests = self.load_tests()
        mutants = self.load_mutants()

        #kills matrix is a dictionary that stores, for each test, the mutants that it kills. This is shared across all mutants
        kills_matrix = {test['test_id']: [] for test in tests}

        #init neural networks
        nn_input_size = 1 + 1 + len(tests)
        self.value_net = ValueNN(nn_input_size)
        self.policy_net = PolicyNN(nn_input_size, len(tests))
        self.observation_net = ObservationNN(nn_input_size + 7)

        print(f"{bcolors.OKBLUE}Starting prioritization{bcolors.ENDC}")

        (rewards, moving_average, v_losses, p_losses, o_losses, moving_average_v_losses, moving_average_p_losses,
         moving_average_o_losses) = [[] for _ in range(8)]

        self.mcts = MCTSAgent(self.policy_net, self.value_net, self.observation_net, tests, kills_matrix,
                              self.sut_name, len(mutants), self.buffer_size, self.batch_size, self.update_delta,
                              self.observation_network_update_delta, self.observation_network_buffer_size, self.rollout_after,
                              self.asymmetric_loss_alpha, self.c_parameter, self.value_network_learning_rate,
                              self.policy_network_learning_rate, self.observation_network_learning_rate)

        #shuffle mutants
        mutants = np.random.permutation(mutants)

        mutant_count = 0

        for index,mutant in enumerate(mutants):

            mutant_count += 1

            no_test_killing = True
            if len(mutant[
                       "testResults"]) > 0:  # for test purposes, we remove the rewards of the mutants that are not killed by any test
                for test_file in mutant["testResults"]:
                    if len(mutant["testResults"][test_file]["failed"]) > 0:
                        no_test_killing = False
                        break

            print(f"{bcolors.HEADER}Processing mutant {index} ({mutant['id']}) out of {len(mutants)}{bcolors.ENDC}")

            (reward, v_loss, p_loss, o_loss, number_of_tests_executed, number_of_tests_executed_on_killable_mutants,
             networks_update_freq) = self.mcts.run(mutant, mutant_count, no_test_killing)

            if not no_test_killing:
                rewards.append(reward)
                moving_average.append(np.mean(rewards))
            if o_loss is not None:
                o_losses.append(o_loss)
                moving_average_o_losses.append(np.mean(o_losses))
            if v_loss is not None:
                v_losses.append(v_loss)
                moving_average_v_losses.append(np.mean(v_losses))
            if p_loss is not None:
                p_losses.append(p_loss)
                moving_average_p_losses.append(np.mean(p_losses))
            total_number_of_tests_executed = number_of_tests_executed

            print(f"{bcolors.OKGREEN}Total number of tests executed so far: {total_number_of_tests_executed}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}Total number of tests executed on killable mutants so far: "
                  f"{number_of_tests_executed_on_killable_mutants}{bcolors.ENDC}")

            if (index + 1) % self.plot_delta == 0:
                plot_mutant_prioritization_results(rewards, moving_average, moving_average_v_losses, moving_average_p_losses,
                                                   moving_average_o_losses, networks_update_freq, self.average_delta,
                                                   self.sut_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network-Guided MCTS test prioritizer for mutation testing')
    parser.add_argument('--tests_folder', type=str, help='Path to the folder containing the tests.')
    parser.add_argument('--mutants', type=str, help='Path to the file containing the mutants.')
    parser.add_argument('--cfg', type=str, nargs='?', help='Path to the dot file containing the control flow graph.')
    parser.add_argument('--ast', type=str, nargs='?', help='Path to the file containing the abstract syntax tree.')
    parser.add_argument('--coverage', type=str, help='Path to the coverage file (test matrix).')
    parser.add_argument("--sut_name", type=str, help="Name of the SUT.")
    parser.add_argument('--plot_delta', type=int, default=30, help='Plot every n steps.')
    parser.add_argument('--average_delta', type=int, default=10, help='Average every n steps.')
    parser.add_argument('--buffer_size', type=int, help='Buffer size for the MCTS agent.')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for the MCTS agent.')
    parser.add_argument('--update_delta', type=int, default=1, help='Update value and policy networks every n steps.')
    parser.add_argument('--observation_network_update_delta', type=int, default=1, help='Update observation network every n steps.')
    parser.add_argument('--observation_network_buffer_size', type=int, default=10, help='Buffer size for the observation network.')
    parser.add_argument('--rollout_delay', type=int, default=45, help='Start relying on networks after n steps. (should be higher than the batch size and the update_delta parameters')
    parser.add_argument('--asymmetric_loss_alpha', type=float, default=6.0, help='Alpha parameter for the asymmetric loss function. A higher value will penalize underestimations of the value network more heavily.')
    parser.add_argument('--c_parameter', type=float, default=1.0, help='C parameter for the UCT algorithm. A higher value will make the algorithm explore more.')
    parser.add_argument('--value_network_learning_rate', type=float, default=0.001, help='Learning rate for the value network.')
    parser.add_argument('--policy_network_learning_rate', type=float, default=0.0001, help='Learning rate for the policy network.')
    parser.add_argument('--observation_network_learning_rate', type=float, default=0.001, help='Learning rate for the observation network.')


    args = parser.parse_args()

    prioritizer = Prioritizer(args.tests_folder, args.mutants, args.sut_name, args.plot_delta, args.average_delta, args.buffer_size,
                              args.batch_size, args.update_delta, args.observation_network_update_delta,
                              args.observation_network_buffer_size, args.rollout_delay, args.asymmetric_loss_alpha, args.c_parameter,
                              args.value_network_learning_rate, args.policy_network_learning_rate, args.observation_network_learning_rate)

    start_time = time.time()
    prioritizer.execute()
    end_time = time.time()

    print(f"{bcolors.OKGREEN}Execution time: {round(end_time - start_time, 2)} seconds{bcolors.ENDC}")

