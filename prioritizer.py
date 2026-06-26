import json
import os
import re
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

import numpy as np

from mcts_agent import MCTSAgent

from utils.logger import bcolors
from utils.plotting import plot_mutant_prioritization_results
from networks.policy_nn import PolicyNN
from networks.value_nn import ValueNN
from agents_manager import AgentsManager, AggregationStrategy
from experiment_tracker import ExperimentTracker, analyze_experiment, analyze_pair_disagreement, \
    analyze_disagreement_vs_reward
from analysis import analyze_committee


class Prioritizer:
    '''
    Test prioritizer for mutation testing guided by the MCTS algorithm.
    For each mutant in a set of mutants, the prioritizer will learn to execute the tests in a
    way that maximizes the chance to kill the mutant as soon as possible.
    '''

    def __init__(self, tests_folder_path, mutants_path, sut_name, plot_delta, average_delta, results_file_name,
                 best_params_file_name, tracker_json_path='experiments/disagreement.json', run_dir='experiments'):
        self.tests_folder_path = tests_folder_path
        self.mutants_path = mutants_path
        self.sut_name = sut_name
        self.plot_delta = plot_delta
        self.average_delta = average_delta
        self.mutants = None
        self.tests = None
        self.execution_id = 0
        self.results_file_name = results_file_name
        self.best_params_file_name = best_params_file_name
        self.tracker_json_path = tracker_json_path
        self.run_dir = run_dir

    def load_mutants(self):
        '''
        Load mutants from the mutations file.
        '''
        logging.info(f"{bcolors.HEADER}Loading mutants from {self.mutants_path}{bcolors.ENDC}")

        with open(self.mutants_path, 'r', encoding='utf-8') as f:
            mutants = json.load(f)
        mutants_list = []
        for contract, contract_mutants in mutants.items():
            for mutant in contract_mutants:
                if 'operator' not in mutant:
                    logging.debug(f"{bcolors.WARNING}Mutant {mutant['id']} has no operator. Skipping...{bcolors.ENDC}")
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

    def _safe_series_mean(self, values):
        """
        Return the mean of a numeric series while ignoring None/NaN values.
        """
        cleaned = []
        for value in values:
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            cleaned.append(value)

        return float(np.mean(cleaned)) if cleaned else np.nan

    def load_tests(self):
        """
        load test methods from the tests files within the test folder
        """

        logging.info(f"{bcolors.HEADER}Loading tests from {self.tests_folder_path}{bcolors.ENDC}")

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

        # remove eventual duplicates
        tests = self.remove_duplicates(tests)

        return tests

    def execute(self, execution_id=0, networks=None, kills_matrix=None, run_idx=None, num_runs=None):
        '''
        Execute the prioritizer.
        '''

        logging.info(f"{bcolors.OKBLUE}Starting prioritization{bcolors.ENDC}")

        sut_tests_execution_time = 0
        total_number_of_tests_executed = 0
        number_of_tests_executed_on_killable_mutants = 0

        # Track experiment-level timing
        global_start_time = None
        if run_idx is not None and num_runs is not None:
            global_start_time = round(time.time() * 1000)

        # get start time in milliseconds
        start_time = round(time.time() * 1000)

        (rewards, moving_average, v_losses, p_losses, moving_average_v_losses, moving_average_p_losses) = [[] for _ in
                                                                                                           range(6)]

        # Track rewards per mutant for statistical analysis
        rewards_per_mutant = []
        avg_divergence_per_mutant = []
        avg_rank_correlation_per_mutant = []

        # Build the shared value net optimizer and loss so the AgentsManager
        # can train the single shared value network centrally.
        import torch
        from mcts_agent import AsymmetricLoss
        shared_value_net = networks["shared_value_net"]
        shared_value_opt = torch.optim.Adam(shared_value_net.parameters(), lr=0.001)
        shared_value_loss_fn = AsymmetricLoss(alpha=6.0)

        agents_manager = AgentsManager(
            self.sut_name, self.tests,
            shared_value_net=shared_value_net,
            shared_value_net_optimizer=shared_value_opt,
            shared_value_loss_fn=shared_value_loss_fn,
        )

        from experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(
            json_path=self.tracker_json_path,
            sut_name=self.sut_name,
            experiment_id=execution_id,
            run_idx=run_idx if run_idx is not None else 0,
            num_mutants=len(self.mutants),
            num_tests=len(self.tests),
            aggregation_strategy=AggregationStrategy.WEIGHTED_MEAN.value,
        )
        tracker.begin_run()

        exploration_mcts = MCTSAgent(networks["exploration"]["policy_net"], networks["exploration"]["value_net"],
                                     self.tests, kills_matrix,
                                     self.sut_name, len(self.mutants), len(self.mutants), 40, 1,
                                     45, 6.0, 3.0, 0.001,
                                     0.0001, agents_manager, agent_key="exploration_proposed_test",
                                     trains_value_net=False)

        exploitation_mcts = MCTSAgent(networks["exploitation"]["policy_net"], networks["exploitation"]["value_net"],
                                      self.tests, kills_matrix,
                                      self.sut_name, len(self.mutants), len(self.mutants), 40, 1,
                                      45, 6.0, 0.5, 0.001,
                                      0.0001, agents_manager, agent_key="exploitation_proposed_test",
                                      trains_value_net=False)

        # The diversity agent optimizes search to select tests maximizing Diversity(t)=1−max(similarity(t,t′)) where t′∈ Executed
        diversity_mcts = MCTSAgent(networks["diversity"]["policy_net"], networks["diversity"]["value_net"],
                                   self.tests, kills_matrix,
                                   self.sut_name, len(self.mutants), len(self.mutants), 40, 1,
                                   45, 6.0, 2.0, 0.001,
                                   0.0001, agents_manager, agent_key="diversity_proposed_test",
                                   diversity_bonus_weight=1.0, trains_value_net=False)

        agents_manager.add_agents([exploration_mcts, exploitation_mcts, diversity_mcts])

        aggregation_strategy = AggregationStrategy.WEIGHTED_MEAN

        networks_update_freq = exploration_mcts.UPDATE_DELTA

        mutant_count = 0

        # shuffle mutants before execution
        # np.random.shuffle(self.mutants)

        for index, mutant in enumerate(self.mutants):

            mutant_count += 1

            no_test_killing = True
            if len(mutant[
                       "testResults"]) > 0:  # for test purposes, we remove the rewards of the mutants that are not killed by any test
                for test_file in mutant["testResults"]:
                    if len(mutant["testResults"][test_file]["failed"]) > 0:
                        no_test_killing = False
                        break

            def format_time(ms):
                seconds = ms // 1000
                minutes = seconds // 60
                hours = minutes // 60
                seconds = seconds % 60
                minutes = minutes % 60

                if hours > 0:
                    return f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    return f"{minutes}m {seconds}s"
                else:
                    return f"{seconds}s"

            current_time = round(time.time() * 1000)
            elapsed_time = current_time - start_time

            progress_str = f"Processing mutant {index} ({mutant['id']}) out of {len(self.mutants)}"

            if index > 0:
                avg_time_per_mutant = elapsed_time / index
                remaining_mutants = len(self.mutants) - index
                estimated_remaining_time = avg_time_per_mutant * remaining_mutants

                progress_str += f" | Elapsed: {format_time(elapsed_time)} | Est. Remaining: {format_time(estimated_remaining_time)}"

            # add total experiments ETA if running multiple experiments
            if run_idx is not None and num_runs is not None:
                total_elapsed = current_time - global_start_time
                current_progress = (run_idx + (index / len(self.mutants))) / num_runs

                if current_progress > 0:
                    total_estimated_time = total_elapsed / current_progress
                    total_remaining_time = total_estimated_time - total_elapsed
                    progress_str += f" [Run {run_idx + 1}/{num_runs}] Total Est. Remaining: {format_time(int(total_remaining_time))}"

            logging.info(f"{bcolors.HEADER}{progress_str}{bcolors.ENDC}")

            tracker.begin_mutant(
                mutant_idx=index,
                mutant_id=mutant["id"],
                operator=mutant["operator"],
                killable=not no_test_killing,
            )

            episode_results = agents_manager.run_episode(mutant, mutant_count, no_test_killing, aggregation_strategy,
                                                         tracker=tracker)

            reward, v_loss, p_loss = episode_results["agent_results"][0]

            avg_divergence_per_mutant.append(self._safe_series_mean(episode_results["avg_sym_kl_over_time"]))
            avg_rank_correlation_per_mutant.append(self._safe_series_mean(episode_results["avg_spearman_over_time"]))

            number_of_tests_executed_on_killable_mutants = agents_manager.number_of_tests_executed_on_killable_mutants

            episode_steps = len(episode_results["avg_sym_kl_over_time"])
            tracker.end_mutant(
                reward=float(reward) if not no_test_killing else None,
                tests_to_kill=episode_steps if not no_test_killing else None,
            )

            if not no_test_killing:
                rewards.append(reward)
                moving_average.append(np.mean(rewards))
                rewards_per_mutant.append(reward)
            else:
                rewards_per_mutant.append(np.nan)
            if v_loss is not None:
                v_losses.append(v_loss)
                moving_average_v_losses.append(np.mean(v_losses))
            if p_loss is not None:
                p_losses.append(p_loss)
                moving_average_p_losses.append(np.mean(p_losses))
            total_number_of_tests_executed = agents_manager.number_of_tests_executed

            logging.debug(
                f"{bcolors.OKGREEN}Total number of tests executed so far: {total_number_of_tests_executed}{bcolors.ENDC}")
            logging.debug(f"{bcolors.OKGREEN}Total number of tests executed on killable mutants so far: "
                          f"{number_of_tests_executed_on_killable_mutants}{bcolors.ENDC}")

            """
            Uncomment the following lines to plot the results every 'plot_delta' mutants.
            if (index + 1) % self.plot_delta == 0:
                plot_mutant_prioritization_results(rewards, moving_average, moving_average_v_losses, moving_average_p_losses,
                                                   moving_average_o_losses, networks_update_freq, self.average_delta,
                                                   self.sut_name)
            """

            if index == len(self.mutants) - 1:
                plots_dir = os.path.join(self.run_dir, 'plots')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)

                plot_mutant_prioritization_results(rewards, moving_average, moving_average_v_losses,
                                                   moving_average_p_losses,
                                                   networks_update_freq, self.average_delta,
                                                   self.sut_name, should_save=True, save_path=plots_dir,
                                                   execution_id=execution_id)
        end_time = round(time.time() * 1000)

        sut_tests_execution_time = agents_manager.current_sut_tests_execution_time

        execution_time = end_time - start_time

        tracker.end_run(
            total_tests_executed=total_number_of_tests_executed,
            total_tests_on_killable=number_of_tests_executed_on_killable_mutants,
            execution_time_ms=execution_time,
        )

        logging.debug(f"{bcolors.OKGREEN}Execution time: {round(execution_time, 2)} seconds{bcolors.ENDC}")

        return {
            "total_number_of_tests_executed": total_number_of_tests_executed,
            "number_of_tests_executed_on_killable_mutants": number_of_tests_executed_on_killable_mutants,
            "rewards": rewards,
            "moving_average": moving_average,
            "v_losses": v_losses,
            "p_losses": p_losses,
            "moving_average_v_losses": moving_average_v_losses,
            "moving_average_p_losses": moving_average_p_losses,
            "execution_time": execution_time,
            "sut_tests_execution_time": sut_tests_execution_time,
            "rewards_per_mutant": rewards_per_mutant,
            "avg_divergence_per_mutant": avg_divergence_per_mutant,
            "avg_rank_correlation_per_mutant": avg_rank_correlation_per_mutant,
        }

    def launch_single_prioritization(self, num_runs=1):
        """
        Execute the prioritizer multiple times on the same mutants and tests with the same parameters.
        Collects results from all runs for statistical analysis. In this branch, we leverage multiple mcts agents
        with different exploration parameters and then aggregate the results before choosing the next test to execute.
        """

        all_runs_rewards = []
        all_runs_divergencies = []
        all_runs_rank_correlations = []
        all_runs_performances = []

        overall_start_time = round(time.time() * 1000)
        run_times = []

        for run_idx in range(num_runs):
            run_start_time = round(time.time() * 1000)

            logging.info(f"{bcolors.HEADER}Starting run {run_idx + 1}/{num_runs}{bcolors.ENDC}")

            # kills matrix is a dictionary that stores, for each test, the mutants that it kills. This is shared across all mutants
            kills_matrix = {test['test_id']: [] for test in self.tests}

            # init neural networks
            nn_input_size = 1 + 1 + len(self.tests)

            # Shared value network: one instance used by all three agents.
            # The value net estimates expected reward given a state, which is a
            # property of the environment, not of any individual agent's strategy.
            # Each agent's distinct behaviour is preserved through its own policy net
            # and its unique UCB hyper-parameters (c, diversity_bonus_weight).
            shared_value_net = ValueNN(nn_input_size)

            # policy networks (one per agent — these encode each agent's distinct strategy)
            exploration_agent_policy_net = PolicyNN(nn_input_size, len(self.tests))
            exploitation_agent_policy_net = PolicyNN(nn_input_size, len(self.tests))
            diversity_agent_policy_net = PolicyNN(nn_input_size, len(self.tests))

            networks = {
                'shared_value_net': shared_value_net,
                'exploration': {
                    'value_net': shared_value_net,
                    'policy_net': exploration_agent_policy_net
                },
                'exploitation': {
                    'value_net': shared_value_net,
                    'policy_net': exploitation_agent_policy_net
                },
                'diversity': {
                    'value_net': shared_value_net,
                    'policy_net': diversity_agent_policy_net
                }
            }

            # Execute prioritizer using these hyperparameters, passing run information for ETA tracking
            performance = self.execute(
                execution_id=self.execution_id + run_idx,
                networks=networks,
                run_idx=run_idx,
                num_runs=num_runs
            )

            run_end_time = round(time.time() * 1000)
            run_duration = run_end_time - run_start_time
            run_times.append(run_duration)

            all_runs_rewards.append(performance["rewards_per_mutant"])
            all_runs_divergencies.append(performance["avg_divergence_per_mutant"])
            all_runs_rank_correlations.append(performance["avg_rank_correlation_per_mutant"])
            all_runs_performances.append(performance)

            logging.debug(f"{bcolors.OKGREEN}Run {run_idx + 1} completed.{bcolors.ENDC}")
            logging.debug(f"Total tests executed: {performance['total_number_of_tests_executed']}")
            logging.debug(
                f"Total tests executed on killable mutants: {performance['number_of_tests_executed_on_killable_mutants']}")

            if run_idx < num_runs - 1:
                def format_time(ms):
                    seconds = ms // 1000
                    minutes = seconds // 60
                    hours = minutes // 60
                    seconds = seconds % 60
                    minutes = minutes % 60

                    if hours > 0:
                        return f"{hours}h {minutes}m {seconds}s"
                    elif minutes > 0:
                        return f"{minutes}m {seconds}s"
                    else:
                        return f"{seconds}s"

                avg_run_time = sum(run_times) / len(run_times)
                remaining_runs = num_runs - (run_idx + 1)
                estimated_remaining_time = avg_run_time * remaining_runs

                logging.debug(f"{bcolors.OKBLUE}Average time per run: {format_time(int(avg_run_time))} | "
                              f"Est. time for remaining {remaining_runs} run(s): {format_time(int(estimated_remaining_time))}{bcolors.ENDC}\n")

        # Plot aggregated results
        from utils.plotting import plot_multiple_runs_results
        plot_multiple_runs_results(
            all_runs_rewards,
            self.sut_name,
            True,
            os.path.join(self.run_dir, 'plots'),
            all_runs_divergencies,
            all_runs_rank_correlations,
        )

        # logging.debug summary statistics
        logging.info(f"\n{bcolors.HEADER}Summary across {num_runs} runs for SUT {self.sut_name}:{bcolors.ENDC}")
        avg_tests = np.mean([p["total_number_of_tests_executed"] for p in all_runs_performances])
        std_tests = np.std([p["total_number_of_tests_executed"] for p in all_runs_performances])
        avg_killable = np.mean([p["number_of_tests_executed_on_killable_mutants"] for p in all_runs_performances])
        std_killable = np.std([p["number_of_tests_executed_on_killable_mutants"] for p in all_runs_performances])

        logging.info(f"Total tests executed: {avg_tests:.2f} ± {std_tests:.2f}")
        logging.info(f"Total tests on killable mutants: {avg_killable:.2f} ± {std_killable:.2f}")

        self.execution_id += num_runs


if __name__ == '__main__':

    logging.info(f"{bcolors.HEADER}Launching experiments...{bcolors.ENDC}")

    aggregation_strategy = "weighted_average"
    sut_name = "thorwallet"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('experiments', f"{sut_name}_{aggregation_strategy}_{timestamp}")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    results_file_name = f'results_{sut_name}_{aggregation_strategy}_{timestamp}.json'
    best_params_file_name = f'best_params_{sut_name}_{aggregation_strategy}_{timestamp}.json'
    tracker_json_path = os.path.join(run_dir, f'disagreement_{sut_name}_{aggregation_strategy}_{timestamp}.json')

    with open(os.path.join(run_dir, results_file_name), 'w') as f:
        json.dump({"executions": []}, f, indent=4)

    # Emptying directory logic removed as a fresh folder is now created for every standalone run.
    plots_dir = os.path.join(run_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    test_folder_path = os.path.join('case_studies', sut_name, 'test')
    mutants_path = os.path.join('sumo_results', sut_name, 'mutations.json')

    prioritizer = Prioritizer(
        test_folder_path, mutants_path, sut_name, 30, 10,
        results_file_name, best_params_file_name,
        tracker_json_path=tracker_json_path, run_dir=run_dir
    )

    logging.info(f"{bcolors.OKBLUE}Executing single prioritization for {sut_name}{bcolors.ENDC}")
    prioritizer.mutants = prioritizer.load_mutants()
    prioritizer.tests = prioritizer.load_tests()
    logging.info(
        f"{bcolors.OKBLUE}Loaded {len(prioritizer.mutants)} mutants and {len(prioritizer.tests)} tests for {sut_name}{bcolors.ENDC}")
    prioritizer.launch_single_prioritization(num_runs=1)  # Run 5 times for statistical significance

    analyze_committee(tracker_json_path)