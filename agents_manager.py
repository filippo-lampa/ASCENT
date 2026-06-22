from enum import Enum
import numpy as np

from mcts_agent import MCTSAgent
from utils.consts import mutant_operators_list


class AggregationStrategy(Enum):
    """
    Aggregation strategies for combining rankings/distributions from multiple
    Neural-MCTS agents (e.g., Explorer, Exploiter, Diversity).

    Each agent provides a ranking (score distribution) over all tests.
    """

    ARITHMETIC_MEAN = "arithmetic_mean"  # Score(t) = (1/N) * Σ p_i(t). Democratic consensus, average score across all agents

    GEOMETRIC_MEAN = "geometric_mean"  # Score(t) = (∏ p_i(t))^(1/N). Strong agreement required, if one agent strongly disagrees, score drops sharply

    WEIGHTED_MEAN = "weighted_mean"  # Score(t) = Σ w_i * p_i(t). Trust agents differently. Weights can be static, adaptive, or based on recent mutant kill performance

    BORDA_COUNT = "borda_count"  # Convert score distributions into rankings and aggregate ranks. Robust rank fusion that avoids calibration issues


class AgentsManager:
    """
    Manages the aggregation of predictions by the managers and executes the chosen test. The three MCTS agents are ran
    in lock-step on each mutant. At every iteration each agent shares a test ranking based on its own UCB scores, which
    are then aggregated by this manager. The resulting test is executed once and applied to all three agents.
    """

    def __init__(self, sut_name, tests):
        self.agents = []
        self.number_of_tests_executed_on_killable_mutants = 0
        self.current_sut_tests_execution_time = 0
        self.number_of_tests_executed = 0
        self.sut_name = sut_name
        self.tests = tests
        # Each agent provides a ranking (score distribution) over all tests
        # step_solutions: {agent_name -> list of floats (scores/probabilities) for each test}
        self.step_solutions = {
            "exploration_proposed_test": [],
            "exploitation_proposed_test": [],
            "diversity_proposed_test": []
        }
        # How many times each test has been actually executed across all mutants.
        # Used by the diversity agent to boost under-executed tests in its UCB score.
        self.test_execution_counts = [0] * len(tests)

    def add_agents(self, agents):
        self.agents.extend(agents)

    def execute_test_on_mutant(self, test, mutant, mutant_not_killable):
        """
        Execute the test against the mutant and return the outcome.
        """

        if not mutant_not_killable:
            self.number_of_tests_executed_on_killable_mutants += 1

        self.number_of_tests_executed += 1

        # Track how many times this specific test has been executed (used by the diversity agent)
        test_index = self.tests.index(test)
        self.test_execution_counts[test_index] += 1

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

        return outcome

    def take_step(self, action, env, mutant, mutant_not_killable):
        """
        Perform all necessary actions on the environment after selecting an action.
        """

        # Execute the test against the mutant
        killed = self.execute_test_on_mutant(self.tests[action], mutant, mutant_not_killable)

        state = MCTSAgent.State(env.test_sequence, mutant_operators_list.index(mutant["operator"]), action)

        terminal_state = False
        if len(env.test_sequence) == len(self.tests) or killed:
            terminal_state = True

        return state, killed, terminal_state

    def share_solution(self, ranking, agent_proposing):
        """
        Store the agent's ranking (score distribution) over all tests.
        """
        if not isinstance(ranking, (list, tuple)):
            raise ValueError(f"Expected ranking to be list/tuple, got {type(ranking)}")
        if len(ranking) != len(self.tests):
            raise ValueError(f"Ranking length {len(ranking)} != number of tests {len(self.tests)}")
        self.step_solutions[agent_proposing] = list(ranking)
        print(f"Agent '{agent_proposing}' shared ranking: {ranking}")

    def reset_step_solutions(self):
        """
        Clear the rankings stored for the current iteration, so that the next
        iteration starts with no proposals.
        """
        self.step_solutions = {agent_key: [] for agent_key in self.step_solutions}

    def _compute_disagreement_metrics(self, proposals):
        """
        Compute pairwise disagreement metrics between agents.

        Returns:
            {
                "pairwise": [{"agent_a", "agent_b", "sym_kl", "spearman"}, ...],
                "avg_sym_kl": float,
                "avg_spearman": float,
            }
        """
        import math
        from itertools import combinations

        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None

        epsilon = 1e-10

        def normalize(scores):
            s = sum(scores)
            if s == 0:
                return [1.0 / len(scores)] * len(scores)
            return [max(float(x) / s, epsilon) for x in scores]

        agent_keys = list(proposals.keys())
        distributions = {k: normalize(proposals[k]) for k in agent_keys}

        pairwise = []
        sym_kl_values = []
        spearman_values = []

        for agent_a, agent_b in combinations(agent_keys, 2):
            p = distributions[agent_a]
            q = distributions[agent_b]

            kl_a_b = sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))
            kl_b_a = sum(q[i] * math.log(q[i] / p[i]) for i in range(len(p)))
            sym_kl = float(kl_a_b + kl_b_a)

            corr = None
            if spearmanr:
                corr, _ = spearmanr(proposals[agent_a], proposals[agent_b])
                if corr is not None and np.isfinite(corr):
                    corr = float(corr)
                else:
                    corr = None

            pairwise.append({
                "agent_a": agent_a,
                "agent_b": agent_b,
                "sym_kl": sym_kl,
                "spearman": corr,
            })
            sym_kl_values.append(sym_kl)
            if corr is not None:
                spearman_values.append(corr)

        return {
            "pairwise": pairwise,
            "avg_sym_kl": float(np.mean(sym_kl_values)) if sym_kl_values else 0.0,
            "avg_spearman": float(np.mean(spearman_values)) if spearman_values else 0.0,
        }

    def _log_disagreement_metrics(self, metrics):

        print("\n--- Committee Disagreement Analysis ---")

        for item in metrics.get("pairwise", []):
            corr = item["spearman"]
            corr_text = f"{corr:.4f}" if isinstance(corr, float) else "scipy required"
            print(f"[{item['agent_a']} vs {item['agent_b']}]:")
            print(f"  -> Symmetric KL Divergence: {item['sym_kl']:.4f}")
            print(f"  -> Spearman Correlation:    {corr_text}")

        print("----------------------------------------\n")

    def run_episode(self, mutant, mutant_number, mutant_not_killable, aggregation_strategy, tracker=None):
        """
        Run a full prioritization episode for a single mutant, driving all the
        registered MCTS agents (exploration, exploitation, diversity) in
        lock-step.

        At every iteration (including the very first one, used to choose the
        first test of the episode):
          1- each agent computes a ranking (score distribution over all tests)
             based on its own UCB scores and shares it via `share_solution`
          2- the rankings are aggregated according to `aggregation_strategy`
             into a single chosen test
          3- that test is the one actually executed against the mutant
          4- each agent updates its own tree/state with the outcome of that
             single execution

        This continues until the mutant is killed or all tests have been
        executed.

        `tracker` is an optional ExperimentTracker instance.  When provided,
        every step's disagreement metrics are recorded automatically.
        """

        for agent in self.agents:
            agent.prepare_episode(mutant, mutant_number, mutant_not_killable)

        avg_sym_kl_over_time = []
        avg_spearman_over_time = []

        # ── root step ─────────────────────────────────────────────────────────
        self.reset_step_solutions()
        for agent in self.agents:
            ranking = agent.propose_root_ranking()
            self.share_solution(ranking, agent.agent_key)

        root_metrics = self._calculate_and_log_disagreement(self._valid_proposals())
        avg_sym_kl_over_time.append(root_metrics["avg_sym_kl"])
        avg_spearman_over_time.append(root_metrics["avg_spearman"])

        chosen_action = self.aggregate_solutions(aggregation_strategy, disagreement_metrics=root_metrics)

        env = MCTSAgent.State([chosen_action], mutant_operators_list.index(mutant["operator"]), chosen_action)
        _, killed, terminal_state = self.take_step(chosen_action, env, mutant, mutant_not_killable)

        for agent in self.agents:
            agent.apply_root(chosen_action, killed, terminal_state)

        if tracker is not None:
            agent_top_picks = {
                key: list(map(int, np.argsort(ranking)[::-1][:3]))
                for key, ranking in self._valid_proposals().items()
            }
            tracker.record_step(
                step_idx=0,
                chosen_test_idx=chosen_action,
                killed=bool(killed),
                metrics=root_metrics,
                agent_top_picks=agent_top_picks,
            )

        done = terminal_state

        # ── subsequent steps ──────────────────────────────────────────────────
        while not done:
            self.reset_step_solutions()
            for agent in self.agents:
                ranking = agent.propose_step_ranking()
                self.share_solution(ranking, agent.agent_key)

            step_metrics = self._calculate_and_log_disagreement(self._valid_proposals())
            avg_sym_kl_over_time.append(step_metrics["avg_sym_kl"])
            avg_spearman_over_time.append(step_metrics["avg_spearman"])

            # Snapshot proposals BEFORE aggregate_solutions clears/uses them
            current_proposals = self._valid_proposals()

            chosen_action = self.aggregate_solutions(aggregation_strategy, disagreement_metrics=step_metrics)

            env = MCTSAgent.State(env.test_sequence + [chosen_action],
                                  mutant_operators_list.index(mutant["operator"]), chosen_action)
            _, killed, terminal_state = self.take_step(chosen_action, env, mutant, mutant_not_killable)

            for agent in self.agents:
                agent.apply_step(chosen_action, killed, terminal_state, env.test_sequence)

            if tracker is not None:
                agent_top_picks = {
                    key: list(map(int, np.argsort(ranking)[::-1][:3]))
                    for key, ranking in current_proposals.items()
                }
                tracker.record_step(
                    step_idx=len(env.test_sequence) - 1,
                    chosen_test_idx=chosen_action,
                    killed=bool(killed),
                    metrics=step_metrics,
                    agent_top_picks=agent_top_picks,
                )

            done = terminal_state

        return {
            "agent_results": [agent.finish_episode() for agent in self.agents],
            "avg_sym_kl_over_time": avg_sym_kl_over_time,
            "avg_spearman_over_time": avg_spearman_over_time,
        }

    def _calculate_and_log_disagreement(self, proposals):
        """
        Computes and logs pairwise divergence and rank correlation between the three agents to analyze committee diversity.
        """
        metrics = self._compute_disagreement_metrics(proposals)
        self._log_disagreement_metrics(metrics)
        return metrics

    def aggregate_solutions(self, aggregation_strategy, disagreement_metrics=None):
        """
        Aggregate the rankings stored in `self.step_solutions` according to
        the selected strategy. Each agent provides a score distribution over all tests. Returns the index of the test
        with the highest aggregated score.
        """
        # Ensure all three agents provided their ranking
        required_agents = {"exploration_proposed_test", "exploitation_proposed_test", "diversity_proposed_test"}
        proposals = self._valid_proposals()
        missing = required_agents - set(proposals.keys())
        print(f"Aggregating solutions with strategy {aggregation_strategy}. Proposals received from agents: {list(proposals.keys())}")
        if missing:
            raise RuntimeError(f"Missing proposals from agents: {sorted(list(missing))}")

        if disagreement_metrics is None:
            self._calculate_and_log_disagreement(proposals)

        if aggregation_strategy == AggregationStrategy.ARITHMETIC_MEAN:
            return self.aggregate_arithmetic_mean()
        elif aggregation_strategy == AggregationStrategy.GEOMETRIC_MEAN:
            return self.aggregate_geometric_mean()
        elif aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN:
            return self.aggregate_weighted_mean()
        elif aggregation_strategy == AggregationStrategy.BORDA_COUNT:
            return self.aggregate_borda_count()
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")

    def _valid_proposals(self):
        """
        Return a dict of agent_key - ranking (list of scores) for valid proposals.
        A proposal is valid if it's a non-empty list with length equal to num_tests.
        """
        valid = {}
        for k, v in self.step_solutions.items():
            if isinstance(v, (list, tuple)) and len(v) == len(self.tests):
                valid[k] = list(v)
        return valid

    def aggregate_arithmetic_mean(self):
        """
        Democratic consensus. Average score across all agents through arithmetic mean. Returns the index of the test
        with the highest mean score. Score(t) = (1/N) * Σ p_i(t)
        """
        proposals = self._valid_proposals()
        if not proposals:
            return None

        n_agents = len(proposals)
        n_tests = len(self.tests)

        # Compute mean score for each test
        scores = [0.0] * n_tests
        for agent_ranking in proposals.values():
            for t_idx, score in enumerate(agent_ranking):
                scores[t_idx] += float(score)

        # Average
        scores = [s / n_agents for s in scores]

        # Return index of highest score (deterministic tie-breaker: lowest index)
        best_idx = scores.index(max(scores))
        print(f"Aggregated scores (arithmetic mean): {scores}, best test index: {best_idx}")
        return best_idx

    def aggregate_geometric_mean(self):
        """
        Strong agreement required. If one agent strongly disagrees, score drops sharply.
        Handles zeros gracefully (clipped to small epsilon to avoid log issues).
        Returns the index of the test with the highest geometric mean. Score(t) = (∏ p_i(t))^(1/N)
        """
        proposals = self._valid_proposals()
        if not proposals:
            return None


        n_agents = len(proposals)
        n_tests = len(self.tests)
        epsilon = 1e-10  # Avoid log(0)

        # Compute geometric mean for each test
        scores = [0.0] * n_tests
        for test_idx in range(n_tests):
            product = 1.0
            for agent_ranking in proposals.values():
                score = max(float(agent_ranking[test_idx]), epsilon)
                product *= score
            # Geometric mean = product^(1/n_agents)
            scores[test_idx] = product ** (1.0 / n_agents)

        # Return index of highest score
        best_idx = scores.index(max(scores))
        print(f"Aggregated scores (geometric mean): {scores}, best test index: {best_idx}")
        return best_idx

    def aggregate_weighted_mean(self):
        """
        Trust agents differently. Weights can be:
        - static: use self.agent_weights if available
        - adaptive: use self.agent_performance if available
        Falls back to equal weights (1.0 each) if neither is provided.

        Returns the index of the test with the highest weighted score. Score(t) = Σ w_i * p_i(t)
        """
        proposals = self._valid_proposals()
        if not proposals:
            return None

        n_tests = len(self.tests)

        # Get weights: try adaptive first, then static, fall back to equal
        agent_weights = getattr(self, "agent_performance", None)
        if agent_weights is None:
            agent_weights = getattr(self, "agent_weights", None)
        if agent_weights is None:
            agent_weights = {}

        # Compute weighted score for each test
        scores = [0.0] * n_tests
        for agent_key, agent_ranking in proposals.items():
            w = float(agent_weights.get(agent_key, 1.0))
            for t_idx, score in enumerate(agent_ranking):
                scores[t_idx] += w * float(score)

        # Return index of highest score
        best_idx = scores.index(max(scores))
        print(f"Aggregated scores (weighted mean): {scores}, best test index: {best_idx}")
        return best_idx

    def aggregate_borda_count(self):
        """
        Convert score distributions into rankings and aggregate ranks. Robust rank fusion that avoids calibration issues.
        For each agent, convert its scores into a ranking (higher score = lower rank number = more points).
        Sum ranks across agents and pick the test with the lowest total rank sum. Returns the index of the test with the
        best (lowest) total rank.
        """
        proposals = self._valid_proposals()
        if not proposals:
            return None

        n_tests = len(self.tests)
        n_agents = len(proposals)

        # Compute rank score for each test (lower is better)
        rank_sums = [0] * n_tests

        for agent_ranking in proposals.values():
            # Sort tests by score (descending), then get ranks for each test
            scored_tests = [(t_idx, float(agent_ranking[t_idx])) for t_idx in range(n_tests)]
            # Create ranking: highest score gets rank 0, second highest gets rank 1, etc.
            scored_tests.sort(key=lambda x: -x[1])  # Sort by score descending

            # Assign ranks
            for rank, (t_idx, _) in enumerate(scored_tests):
                rank_sums[t_idx] += rank  # Lower rank sums are better

        # Return index of lowest rank sum (best consensus)
        best_idx = rank_sums.index(min(rank_sums))
        print(f"Aggregated rank sums (Borda count): {rank_sums}, best test index: {best_idx}")
        return best_idx