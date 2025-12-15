import argparse
import itertools
import json
import os
import re
import time

import optuna

import numpy as np

from mcts_agent import MCTSAgent

from utils.logger import bcolors
from utils.plotting import plot_mutant_prioritization_results
from utils.consts import baseline_results_per_project, suts_names
from networks.policy_nn import PolicyNN
from networks.value_nn import ValueNN

class Prioritizer:
    '''
    Test prioritizer for mutation testing guided by the MCTS algorithm.
    For each mutant in a set of mutants, the prioritizer will learn to execute the tests in a
    way that maximizes the chance to kill the mutant as soon as possible.
    '''
    def __init__(self, tests_folder_path, mutants_path, sut_name, plot_delta, average_delta, results_file_name, best_params_file_name):
        self.tests_folder_path = tests_folder_path
        self.mutants_path = mutants_path
        self.sut_name = sut_name
        self.plot_delta = plot_delta
        self.average_delta = average_delta
        self.mutants = None
        self.tests = None
        self.execution_id = 0
        self.parameters_set_id = 0
        self.results_file_name = results_file_name
        self.best_params_file_name = best_params_file_name

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

    def execute(self, execution_id=0, parameters_set_id=0, buffer_size=None, batch_size=None,
                update_delta=None, rollout_after=None, asymmetric_loss_alpha=None, c_parameter=None, value_lr=None,
                policy_lr=None, policy_net=None, value_net=None,
                kills_matrix=None, inference_batch_size=128, window_size=32):
        '''
        Execute the prioritizer on all mutants.

        Parameters:
        - window_size: Number of mutants to process concurrently before batching inferences
        - inference_batch_size: Maximum number of inference requests to process at once (GPU memory limit)
        '''

        print(f"{bcolors.OKBLUE}Starting prioritization for {len(self.mutants)} mutants{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Window size: {window_size} mutants{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Inference batch size: {inference_batch_size} requests{bcolors.ENDC}")

        start_time = round(time.time() * 1000)

        # Initialize MCTS agent
        mcts = MCTSAgent(policy_net, value_net, self.tests, kills_matrix,
                        self.sut_name, len(self.mutants), buffer_size, batch_size, update_delta,
                        rollout_after, asymmetric_loss_alpha, c_parameter, value_lr, policy_lr,
                        inference_batch_size, window_size)

        # Run on all mutants - MCTS handles parallelization internally
        (avg_reward, v_loss, p_loss, num_tests_executed, num_tests_on_killable,
         networks_update_freq, sut_tests_execution_time) = mcts.run(self.mutants)

        # Prepare results
        rewards = [avg_reward]
        moving_average = [avg_reward]
        v_losses = [v_loss] if v_loss is not None else []
        p_losses = [p_loss] if p_loss is not None else []
        moving_average_v_losses = [v_loss] if v_loss is not None else []
        moving_average_p_losses = [p_loss] if p_loss is not None else []

        print(f"{bcolors.OKGREEN}Total tests executed: {num_tests_executed}{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Tests executed on killable mutants: {num_tests_on_killable}{bcolors.ENDC}")

        # Plot results
        if not os.path.exists('experiments/plots'):
            os.makedirs('experiments/plots')

        plot_mutant_prioritization_results(rewards, moving_average, moving_average_v_losses, moving_average_p_losses,
                                           networks_update_freq, self.average_delta, self.sut_name, should_save=True,
                                           save_path='experiments/plots', execution_id=execution_id, parameters_set_id=parameters_set_id)

        end_time = round(time.time() * 1000)
        execution_time = end_time - start_time

        print(f"{bcolors.OKGREEN}Execution time: {execution_time} milliseconds{bcolors.ENDC}")

        return (num_tests_executed, num_tests_on_killable, rewards, moving_average,
                v_losses, p_losses, moving_average_v_losses, moving_average_p_losses,
                execution_time, sut_tests_execution_time)

    def objective(self, trial):

        all_params = ['c_parameter', 'asymmetric_loss_alpha', 'rollout_after',
                      'update_delta', 'buffer_size', 'batch_size']

        chosen = trial.suggest_categorical("chosen_params", list(itertools.combinations(all_params, 5)))

        params = {
            'value_network_learning_rate': 0.001,
            'policy_network_learning_rate': 0.0001,
        }

        defaults = {
            'c_parameter': 1.0,
            'batch_size': 40,
            'asymmetric_loss_alpha': 6.0,
            'rollout_after': 45,
            'update_delta': 1,
            'buffer_size': len(self.mutants)
        }

        for name in all_params:
            if name in chosen:
                if name == 'c_parameter':
                    params[name] = trial.suggest_categorical(name, [0.1, 1.0, 2.0, 3.0, 5.0])
                elif name == 'batch_size':
                    params[name] = trial.suggest_categorical(name, [16, 32, 40, 64, 128])
                elif name == 'asymmetric_loss_alpha':
                    params[name] = trial.suggest_categorical(name, [1.0, 3.25, 5.5, 7.75, 10.0])
                elif name == 'rollout_after':
                    params[name] = trial.suggest_categorical(name, [0, 25, 50, 75, 100])
                elif name == 'update_delta':
                    params[name] = trial.suggest_categorical(name, [1, 3, 5, 7, 10])
                elif name == 'buffer_size':
                    params[name] = trial.suggest_categorical(name, [int(len(self.mutants) * factor) for factor in [0.2, 0.4, 0.6, 0.8, 1.0]])
            else:
                params[name] = defaults[name]

        kills_matrix = {test['test_id']: [] for test in self.tests}

        #init neural networks
        nn_input_size = 1 + 1 + len(self.tests)
        value_net = ValueNN(nn_input_size)
        policy_net = PolicyNN(nn_input_size, len(self.tests))

        # Execute prioritizer
        performance = self.execute(
            execution_id=self.execution_id,
            parameters_set_id=self.parameters_set_id,
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size'],
            update_delta=params['update_delta'],
            rollout_after=params['rollout_after'],
            asymmetric_loss_alpha=params['asymmetric_loss_alpha'],
            c_parameter=params['c_parameter'],
            value_lr=params['value_network_learning_rate'],
            policy_lr=params['policy_network_learning_rate'],
            policy_net=policy_net,
            value_net=value_net,
            kills_matrix=kills_matrix,
            inference_batch_size=128,
            window_size=32
        )

        # Save results
        result = {
            "execution_id": self.execution_id,
            "sut_name": self.sut_name,
            "parameters": {
                "buffer_size": params["buffer_size"],
                "batch_size": params["batch_size"],
                "update_delta": params["update_delta"],
                "rollout_after": params["rollout_after"],
                "asymmetric_loss_alpha": params["asymmetric_loss_alpha"],
                "c_parameter": params["c_parameter"],
                "value_network_learning_rate": params["value_network_learning_rate"],
                "policy_network_learning_rate": params["policy_network_learning_rate"]
            },
            "total_tests_executed": performance[0],
            "baseline_total_tests_executed": baseline_results_per_project[self.sut_name + '_baseline_total_tests_executed'],
            "total_tests_executed_on_killable_mutants": performance[1],
            "baseline_total_tests_executed_on_killable_mutants": baseline_results_per_project[self.sut_name + '_baseline_total_tests_executed_on_killable_mutants'],
            "percentual_improvement_on_killable_mutants": (
                (baseline_results_per_project[self.sut_name + '_baseline_total_tests_executed_on_killable_mutants'] - performance[1]) /
                baseline_results_per_project[self.sut_name + '_baseline_total_tests_executed_on_killable_mutants']) * 100
                if baseline_results_per_project[self.sut_name + '_baseline_total_tests_executed_on_killable_mutants'] > 0 else 0,
            "average_number_of_tests_needed_to_kill_a_mutant": np.mean([len(test['test_id']) for test in self.tests]),
            "execution_time": performance[8],
            "tests_execution_time": performance[9],
            "total_execution_time": performance[8] + performance[9]
        }

        #update results json file
        with open('experiments/' + self.results_file_name, 'r+') as f:
            results = json.load(f)
            results['executions'].append(result)
            f.seek(0)
            json.dump(results, f, indent=4)


        self.execution_id += 1

        return performance[0]  # Return the total number of tests executed as the objective value for minimization

    def launch_experiments_optuna(self):
        """
        Execute the prioritizer multiple times on the same mutants and tests, with different parameters selected through grid search.
        """

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())
        study.optimize(self.objective, n_trials=500)

        #print best parameters for the SUT in the best_params file in the experiments folder
        print("Best parameters:", study.best_params)
        with open('experiments/' + self.best_params_file_name, 'w') as f:
            json.dump(study.best_params, f, indent=4)

        # retrieve the relative execution from the json (check the entry with the same parameters and return its execution id)
        with open('experiments/' + self.results_file_name, 'r') as f:
            results = json.load(f)
            executions = results.get("executions", [])
            for execution in executions:
                if execution["parameters"] == study.best_params:
                    print(f"Best execution ID: {execution['execution_id']}")
                    break
            else:
                print("No matching execution found for the best parameters.")

    def launch_single_experiment_default_params(self):
        """
        Execute the prioritizer once on the same mutants and tests, with default parameters.
        """

        kills_matrix = {test['test_id']: [] for test in self.tests}

        #init neural networks
        nn_input_size = 1 + 1 + len(self.tests)
        value_net = ValueNN(nn_input_size)
        policy_net = PolicyNN(nn_input_size, len(self.tests))

        # Execute prioritizer with optimized parameters
        self.execute(
            execution_id=self.execution_id,
            parameters_set_id=self.parameters_set_id,
            buffer_size=len(self.mutants),  # Mantieni tutta l'esperienza
            batch_size=40,
            update_delta=1,  # Aggiorna le reti ad ogni mutante completato
            rollout_after=60,  # Riduci da 45 a 20 - inizia prima a usare le NN
            asymmetric_loss_alpha=6.0,
            c_parameter=0.1,
            value_lr=0.001,
            policy_lr=0.0001,
            policy_net=policy_net,
            value_net=value_net,
            kills_matrix=kills_matrix,
            inference_batch_size=64,  # Riduci da 256 a 64 - batch più piccoli, inferenze più frequenti
            window_size=16  # Riduci da 64 a 16 - processa meno mutanti alla volta per sfruttare meglio la conoscenza
        )



if __name__ == '__main__':

    if not os.path.exists('experiments'):
        os.makedirs('experiments')


    next_experiment_id = 0
    if os.path.exists('experiments/results.json'):
        existing_files = [f for f in os.listdir('experiments') if f.startswith('results')]
    experiment_ids = [int(re.search(r'results_(\d+)\.json', f).group(1)) for f in existing_files if
                      re.search(r'results_(\d+)\.json', f)]
    next_experiment_id = max(experiment_ids) + 1 if experiment_ids else 1

    results_file_name = 'results.json' if next_experiment_id == 0 else f'results_{next_experiment_id}.json'

    next_best_params_id = 0
    if os.path.exists('experiments/best_params.json'):
        existing_files = [f for f in os.listdir('experiments') if f.startswith('best_params')]
    best_params_ids = [int(re.search(r'best_params_(\d+)\.json', f).group(1)) for f in existing_files if
                       re.search(r'best_params_(\d+)\.json', f)]
    next_best_params_id = max(best_params_ids) + 1 if best_params_ids else 1

    best_params_file_name = 'best_params.json' if next_best_params_id == 0 else f'best_params_{next_best_params_id}.json'

    with open('experiments/' + results_file_name, 'w') as f:
        json.dump({"executions": []}, f, indent=4)

    #empty the experiments/plots folder
    if not os.path.exists('experiments/plots'):
        os.makedirs('experiments/plots')
    else:
        for file in os.listdir('experiments/plots'):
            file_path = os.path.join('experiments/plots', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    #execute the prioritizer on each project
    parser = argparse.ArgumentParser(description='Network-Guided MCTS test prioritizer for mutation testing')
    parser.add_argument('--sut_name', type=str, help='Path to the folder containing the tests.')

    args = parser.parse_args()
    sut_name = args.sut_name

    test_folder_path = os.path.join('case_studies', sut_name, 'test')
    mutants_path = os.path.join('sumo_results', sut_name, 'mutations.json')
    prioritizer = Prioritizer(test_folder_path, mutants_path, sut_name, 30, 10, results_file_name, best_params_file_name)
    print(f"{bcolors.OKBLUE}Executing prioritizer for {sut_name}{bcolors.ENDC}")
    prioritizer.mutants = prioritizer.load_mutants()
    prioritizer.tests = prioritizer.load_tests()
    print(f"{bcolors.OKBLUE}Loaded {len(prioritizer.mutants)} mutants and {len(prioritizer.tests)} tests for {sut_name}{bcolors.ENDC}")
    prioritizer.launch_single_experiment_default_params()
