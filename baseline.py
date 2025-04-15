import argparse
import json
import os
import re
import sys

import numpy as np

from utils.logger import bcolors


class Baseline:
    """
    This class is used to check how many tests are needed to kill each mutant in the project without prioritization.
    """
    def __init__(self, tests_folder_path, mutants_path):
        self.tests_folder_path = tests_folder_path
        self.mutants_path = mutants_path

    def load_mutants(self):
        '''
        Load mutants from a file.
        '''
        print(f"{bcolors.HEADER}Loading mutants from {self.mutants_path}{bcolors.ENDC}")

        with open(self.mutants_path, 'r', encoding='utf-8') as f:
            mutants = json.load(f)
        mutants_list = []
        for contract, contract_mutants in mutants.items():
            for mutant in contract_mutants:
                mutant['contract'] = contract
                mutants_list.append(mutant)
        return mutants_list


    def remove_duplicates(self, tests):
        test_ids = [test['test_id'] for test in tests]
        duplicates = [test_id for test_id in test_ids if test_ids.count(test_id) > 1]
        if duplicates:
            for test in tests:
                if test['test_id'] in duplicates:
                    tests.remove(test)
                    duplicates.remove(test['test_id'])
        return tests


    def load_tests(self):
        """
        create an array with test path and test name by reading the tests in the folder at self.tests_folder_path
        """
        print(f"{bcolors.HEADER}Loading tests from {self.tests_folder_path}{bcolors.ENDC}")

        test_files_counter = 0

        test_method_names_regex = re.compile(r'it\([\'\"](.*)[\'\"]')
        tests = []
        for root, _, files in os.walk(self.tests_folder_path):
            for test_file in files:
                test_files_counter += 1
                test_file_path = os.path.join(root, test_file)
                if not os.path.isfile(test_file_path) or not(test_file.endswith('.ts') or test_file.endswith('.js')):
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

        test_methods_counter = len(tests)
        print(f"Found {test_files_counter} test files and {test_methods_counter} test methods")
        return tests


    def execute(self, mutants, tests, sut_name):
        """
        Execute each test for each mutant and print the moving average of the number of tests needed to kill each mutant
        """
        #we iterate over the mutants and, following the order of the tests, we check if the mutant is killed by the test
        #until we find a test that kills the mutant.

        total_tests_executed = 0
        total_tests_executed_on_killable_mutants = 0
        tests_needed_list = []

        for mutant in mutants:
            no_test_killing = True
            if len(mutant[
                       "testResults"]) > 0:  # for test purposes, we remove the rewards of the mutants that are not killed by any test
                for test_file in mutant["testResults"]:
                    if len(mutant["testResults"][test_file]["failed"]) > 0:
                        no_test_killing = False
                        break

            tests_executed = 0
            killed = False

            for test in tests:

                if not no_test_killing:
                    total_tests_executed_on_killable_mutants += 1

                tests_executed += 1
                total_tests_executed += 1

                if len(mutant["testResults"]) > 0:
                    if "\\" in list(mutant["testResults"].keys())[0] and "/" in test["test_file_path"]:
                        test_relative_path = test["test_file_path"].split(sut_name)[1].replace("/", "\\")
                    elif "/" in list(mutant["testResults"].keys())[0] and "\\" in test["test_file_path"]:
                        test_relative_path = test["test_file_path"].split(sut_name)[1].replace("\\", "/")
                    else:
                        test_relative_path = test["test_file_path"].split(sut_name)[1]
                    for killing_test_method in mutant["testResults"][test_relative_path]["failed"]:
                        if killing_test_method["title"] == test["test_method_name"]:
                            print(f"Mutant {mutant['id']} killed by test {test['test_id']} after {tests_executed} tests")
                            tests_needed_list.append(tests_executed)
                            killed = True
                            break
                if killed:
                    break

        print(f"Total tests executed: {total_tests_executed} on a total of {len(mutants)} mutants")
        print(f"Total tests executed on killable mutants: {total_tests_executed_on_killable_mutants}")
        print(f"Average number of tests needed to kill a mutant: {np.mean(tests_needed_list)}")

        return total_tests_executed, total_tests_executed_on_killable_mutants


    def execute_multiple(self, mutants, tests, sut_name, n_times):
        """
        Execute each test for each mutant, shuffle tests and mutants and repeat n_times saving the average number of tests needed to kill each mutant
        """
        number_of_tests_executed_list = []
        number_of_tests_on_killable_mutants_list = []
        for i in range(n_times):
            print(f"Execution {i+1}")
            number_tests_executed, number_of_tests_on_killable_mutants = self.execute(np.random.permutation(mutants), np.random.permutation(tests), sut_name)
            number_of_tests_executed_list.append(number_tests_executed)
            number_of_tests_on_killable_mutants_list.append(number_of_tests_on_killable_mutants)
        print(f"Average number of tests executed over {n_times} executions: {np.mean(number_of_tests_executed_list)}")
        print(f"Average number of tests executed on killable mutants over {n_times} executions: {np.mean(number_of_tests_on_killable_mutants_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline test execution on mutants.')
    parser.add_argument('--tests_folder', type=str, help='Path to the folder containing the tests.')
    parser.add_argument('--mutants', type=str, help='Path to the file containing the mutants.')
    parser.add_argument('--sut_name', type=str, help='Name of the SUT.')
    args = parser.parse_args()

    executor = Baseline(args.tests_folder, args.mutants)
    #executor.execute(executor.load_mutants(), executor.load_tests(), args.sut_name)
    executor.execute_multiple(executor.load_mutants(), executor.load_tests(), args.sut_name, 10)

