"""
experiment_tracker.py

Utilities for saving and analysing per-step disagreement metrics produced by
AgentsManager.  Every call to run_episode() generates structured data; this
module writes it to a JSON file and provides aggregate analysis helpers.

JSON layout
-----------
{
  "meta": {
    "sut_name": str,
    "created_at": ISO-8601 str,
    "schema_version": "1.0"
  },
  "experiments": [          # one entry per launch_single_prioritization() call
    {
      "experiment_id": int,
      "run_idx": int,         # 0-based index within the experiment batch
      "sut_name": str,
      "started_at": ISO-8601 str,
      "finished_at": ISO-8601 str | null,
      "num_mutants": int,
      "num_tests": int,
      "aggregation_strategy": str,
      "mutants": [          # one entry per mutant processed in this run
        {
          "mutant_idx": int,
          "mutant_id": str | int,
          "operator": str,
          "killable": bool,
          "steps": [        # one entry per test-execution step inside the episode
            {
              "step_idx": int,                # 0 = root selection
              "chosen_test_idx": int,
              "killed": bool,
              "pairwise": [                   # all agent pairs
                {
                  "agent_a": str,
                  "agent_a_top3": [int, int, int],   # top-3 test indices by this agent's ranking
                  "agent_b": str,
                  "agent_b_top3": [int, int, int],
                  "sym_kl": float,
                  "spearman": float | null
                }
              ],
              "avg_sym_kl": float,
              "avg_spearman": float | null
            }
          ],
          "episode_avg_sym_kl": float | null,
          "episode_avg_spearman": float | null,
          "reward": float | null,             # null for unkillable mutants
          "tests_to_kill": int | null         # null for unkillable mutants
        }
      ],
      "summary": {          # aggregated over all mutants in this run
        "mean_sym_kl": float,
        "std_sym_kl": float,
        "mean_spearman": float | null,
        "std_spearman": float | null,
        "mean_reward": float | null,
        "std_reward": float | null,
        "mean_tests_to_kill": float | null,
        "std_tests_to_kill": float | null,
        "total_tests_executed": int,
        "total_tests_on_killable": int,
        "execution_time_ms": int | null
      }
    }
  ]
}
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any
import logging

import numpy as np

SCHEMA_VERSION = "1.1"

# Internal helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mean(values: list) -> float | None:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(cleaned)) if cleaned else None


def _safe_std(values: list) -> float | None:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.std(cleaned)) if len(cleaned) > 1 else (0.0 if len(cleaned) == 1 else None)


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"meta": {}, "experiments": []}


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


class ExperimentTracker:
    """
    Accumulates disagreement data for a single run, then flushes to JSON.

    Usage (inside Prioritizer.execute):
        tracker = ExperimentTracker(json_path, sut_name, experiment_id, run_idx,
                                    num_mutants, num_tests, aggregation_strategy)
        tracker.begin_run()

        for mutant in mutants:
            tracker.begin_mutant(mutant_idx, mutant_id, operator, killable)

            # root step
            tracker.record_step(step_idx=0,
                                 chosen_test_idx=chosen_action,
                                 killed=bool(killed),
                                 metrics=root_metrics)

            while not done:
                tracker.record_step(step_idx=step_idx,
                                     chosen_test_idx=chosen_action,
                                     killed=bool(killed),
                                     metrics=step_metrics)

            tracker.end_mutant(reward=reward, tests_to_kill=tests_to_kill)

        tracker.end_run(total_tests_executed, total_tests_on_killable, execution_time_ms)
    """

    def __init__(
        self,
        json_path: str,
        sut_name: str,
        experiment_id: int,
        run_idx: int,
        num_mutants: int,
        num_tests: int,
        aggregation_strategy: str,
    ) -> None:
        self.json_path = json_path
        self.sut_name = sut_name
        self.experiment_id = experiment_id
        self.run_idx = run_idx
        self.num_mutants = num_mutants
        self.num_tests = num_tests
        self.aggregation_strategy = aggregation_strategy

        self._run: dict | None = None
        self._current_mutant: dict | None = None

    # Run-level lifecycle

    def begin_run(self) -> None:
        self._run = {
            "experiment_id": self.experiment_id,
            "run_idx": self.run_idx,
            "sut_name": self.sut_name,
            "started_at": _now_iso(),
            "finished_at": None,
            "num_mutants": self.num_mutants,
            "num_tests": self.num_tests,
            "aggregation_strategy": self.aggregation_strategy,
            "mutants": [],
            "summary": None,
        }

    def end_run(
        self,
        total_tests_executed: int,
        total_tests_on_killable: int,
        execution_time_ms: int | None = None,
    ) -> None:
        assert self._run is not None, "Call begin_run() first."

        self._run["finished_at"] = _now_iso()

        # Build run-level summary
        mutants = self._run["mutants"]

        kl_vals = [m["episode_avg_sym_kl"] for m in mutants if m["episode_avg_sym_kl"] is not None]
        sp_vals = [m["episode_avg_spearman"] for m in mutants if m["episode_avg_spearman"] is not None]
        reward_vals = [m["reward"] for m in mutants if m["reward"] is not None]
        ttk_vals = [m["tests_to_kill"] for m in mutants if m["tests_to_kill"] is not None]

        # Aggregate agreement rates across all mutant episodes
        def _mean_agreement_rate(level: str) -> float | None:
            vals = [
                m["agreement_rates"][level]
                for m in mutants
                if m.get("agreement_rates") and m["agreement_rates"][level] is not None
            ]
            return _safe_mean(vals)

        self._run["summary"] = {
            "mean_sym_kl": _safe_mean(kl_vals),
            "std_sym_kl": _safe_std(kl_vals),
            "mean_spearman": _safe_mean(sp_vals),
            "std_spearman": _safe_std(sp_vals),
            "mean_reward": _safe_mean(reward_vals),
            "std_reward": _safe_std(reward_vals),
            "mean_tests_to_kill": _safe_mean(ttk_vals),
            "std_tests_to_kill": _safe_std(ttk_vals),
            "total_tests_executed": total_tests_executed,
            "total_tests_on_killable": total_tests_on_killable,
            "execution_time_ms": execution_time_ms,
            "mean_agreement_rate_full":    _mean_agreement_rate("full"),
            "mean_agreement_rate_soft":    _mean_agreement_rate("soft"),
            "mean_agreement_rate_partial": _mean_agreement_rate("partial"),
            "mean_agreement_rate_none":    _mean_agreement_rate("none"),
        }

        self._flush()

    # Mutant-level lifecycle

    def begin_mutant(
        self,
        mutant_idx: int,
        mutant_id: Any,
        operator: str,
        killable: bool,
    ) -> None:
        self._current_mutant = {
            "mutant_idx": mutant_idx,
            "mutant_id": str(mutant_id),
            "operator": operator,
            "killable": killable,
            "steps": [],
            "episode_avg_sym_kl": None,
            "episode_avg_spearman": None,
            "reward": None,
            "tests_to_kill": None,
        }

    def record_step(
        self,
        step_idx: int,
        chosen_test_idx: int,
        killed: bool,
        metrics: dict,
        agent_top_picks: dict[str, list[int]] | None = None,
    ) -> None:
        """
        Record one test-execution step.

        `metrics` is the dict returned by AgentsManager._compute_disagreement_metrics().
        `agent_top_picks` is a dict mapping agent_key -> top-3 test indices (descending
        rank order), computed from each agent's ranking vector via argsort.
        """
        assert self._current_mutant is not None, "Call begin_mutant() first."

        agent_top_picks = agent_top_picks or {}

        pairwise = [
            {
                "agent_a": p["agent_a"],
                "agent_a_top3": agent_top_picks.get(p["agent_a"], []),
                "agent_b": p["agent_b"],
                "agent_b_top3": agent_top_picks.get(p["agent_b"], []),
                "sym_kl": p["sym_kl"],
                "spearman": p.get("spearman"),
            }
            for p in metrics.get("pairwise", [])
        ]

        # Classify committee agreement for this step:
        #   full      — all agents ranked chosen_test_idx first
        #   soft      — chosen_test_idx appears in every agent's top-3 (but not all #1)
        #   partial   — chosen_test_idx appears in at least one (but not all) top-3 lists
        #   none      — no agent had chosen_test_idx in their top-3
        all_top3s = list(agent_top_picks.values())
        if all_top3s:
            in_top3 = [chosen_test_idx in t3 for t3 in all_top3s]
            ranked_first = [t3[0] == chosen_test_idx for t3 in all_top3s if t3]
            if all(ranked_first):
                agreement = "full"
            elif all(in_top3):
                agreement = "soft"
            elif any(in_top3):
                agreement = "partial"
            else:
                agreement = "none"
        else:
            agreement = None

        self._current_mutant["steps"].append(
            {
                "step_idx": step_idx,
                "chosen_test_idx": chosen_test_idx,
                "killed": killed,
                "agreement": agreement,
                "pairwise": pairwise,
                "avg_sym_kl": metrics.get("avg_sym_kl"),
                "avg_spearman": metrics.get("avg_spearman"),
            }
        )

    def end_mutant(
        self,
        reward: float | None,
        tests_to_kill: int | None,
    ) -> None:
        assert self._current_mutant is not None, "Call begin_mutant() first."

        steps = self._current_mutant["steps"]
        kl_over_steps = [s["avg_sym_kl"] for s in steps if s["avg_sym_kl"] is not None]
        sp_over_steps = [s["avg_spearman"] for s in steps if s["avg_spearman"] is not None]

        self._current_mutant["episode_avg_sym_kl"] = _safe_mean(kl_over_steps)
        self._current_mutant["episode_avg_spearman"] = _safe_mean(sp_over_steps)
        self._current_mutant["reward"] = reward
        self._current_mutant["tests_to_kill"] = tests_to_kill

        # Agreement rates over the episode (only steps where agreement was classified)
        classified = [s["agreement"] for s in steps if s.get("agreement") is not None]
        n = len(classified)
        self._current_mutant["agreement_rates"] = {
            "full":    round(classified.count("full")    / n, 4) if n else None,
            "soft":    round(classified.count("soft")    / n, 4) if n else None,
            "partial": round(classified.count("partial") / n, 4) if n else None,
            "none":    round(classified.count("none")    / n, 4) if n else None,
            "n_steps": n,
        }

        self._run["mutants"].append(self._current_mutant)
        self._current_mutant = None

    # Persistence

    def _flush(self) -> None:
        data = _load_json(self.json_path)

        # Ensure meta block exists
        if not data.get("meta"):
            data["meta"] = {
                "sut_name": self.sut_name,
                "created_at": _now_iso(),
                "schema_version": SCHEMA_VERSION,
            }

        data["experiments"].append(self._run)
        _save_json(self.json_path, data)


def analyze_experiment(json_path: str, experiment_id: int | None = None) -> dict:
    """
    Analyse a single run or all runs for one experiment_id.

    If experiment_id is None, analyses ALL experiments in the file.

    Returns a nested dict:
    {
        "by_experiment": {
            <experiment_id>: {
                "by_run": {<run_idx>: {...stats...}},
                "across_runs": {...stats...}
            }
        },
        "global": {...stats...}
    }
    """
    data = _load_json(json_path)
    experiments = data.get("experiments", [])

    if experiment_id is not None:
        experiments = [e for e in experiments if e["experiment_id"] == experiment_id]

    if not experiments:
        logging.info("No matching experiments found.")
        return {}

    by_experiment: dict = {}

    all_kl: list = []
    all_sp: list = []
    all_rewards: list = []
    all_ttk: list = []

    for exp in experiments:
        eid = exp["experiment_id"]
        rid = exp["run_idx"]

        if eid not in by_experiment:
            by_experiment[eid] = {"by_run": {}, "across_runs": None}

        run_stats = _summarise_run(exp)
        by_experiment[eid]["by_run"][rid] = run_stats

        all_kl.append(run_stats["mean_sym_kl"])
        if run_stats["mean_spearman"] is not None:
            all_sp.append(run_stats["mean_spearman"])
        if run_stats["mean_reward"] is not None:
            all_rewards.append(run_stats["mean_reward"])
        if run_stats["mean_tests_to_kill"] is not None:
            all_ttk.append(run_stats["mean_tests_to_kill"])

    # Across-runs aggregation per experiment_id
    for eid, payload in by_experiment.items():
        runs = list(payload["by_run"].values())
        payload["across_runs"] = _aggregate_run_stats(runs)

    # Global aggregation
    result = {
        "by_experiment": by_experiment,
        "global": {
            "num_experiments": len(by_experiment),
            "num_runs_total": len(experiments),
            "mean_sym_kl": _safe_mean(all_kl),
            "std_sym_kl": _safe_std(all_kl),
            "mean_spearman": _safe_mean(all_sp),
            "std_spearman": _safe_std(all_sp),
            "mean_reward": _safe_mean(all_rewards),
            "std_reward": _safe_std(all_rewards),
            "mean_tests_to_kill": _safe_mean(all_ttk),
            "std_tests_to_kill": _safe_std(all_ttk),
        },
    }

    _print_analysis(result)
    return result


def analyze_pair_disagreement(json_path: str, experiment_id: int | None = None) -> dict:
    """
    Break down avg KL divergence and Spearman correlation per agent *pair*
    across experiments/runs.

    Returns:
    {
        "<agent_a> vs <agent_b>": {
            "mean_sym_kl": float,
            "std_sym_kl": float,
            "mean_spearman": float | null,
            "std_spearman": float | null,
            "num_steps": int        # total steps contributing to this pair
        },
        ...
    }
    """
    data = _load_json(json_path)
    experiments = data.get("experiments", [])

    if experiment_id is not None:
        experiments = [e for e in experiments if e["experiment_id"] == experiment_id]

    pair_buckets: dict[str, dict[str, list]] = {}

    for exp in experiments:
        for mutant in exp.get("mutants", []):
            for step in mutant.get("steps", []):
                for pw in step.get("pairwise", []):
                    key = f"{pw['agent_a']} vs {pw['agent_b']}"
                    if key not in pair_buckets:
                        pair_buckets[key] = {"sym_kl": [], "spearman": []}
                    pair_buckets[key]["sym_kl"].append(pw["sym_kl"])
                    if pw.get("spearman") is not None:
                        pair_buckets[key]["spearman"].append(pw["spearman"])

    result = {}
    for key, bucket in pair_buckets.items():
        result[key] = {
            "mean_sym_kl": _safe_mean(bucket["sym_kl"]),
            "std_sym_kl": _safe_std(bucket["sym_kl"]),
            "mean_spearman": _safe_mean(bucket["spearman"]) if bucket["spearman"] else None,
            "std_spearman": _safe_std(bucket["spearman"]) if bucket["spearman"] else None,
            "num_steps": len(bucket["sym_kl"]),
        }

    logging.debug("\n--- Per-pair disagreement analysis ---")
    for key, stats in result.items():
        sp = f"{stats['mean_spearman']:.4f} ± {stats['std_spearman']:.4f}" if stats["mean_spearman"] is not None else "n/a"
        logging.debug(
            f"  {key}\n"
            f"    KL:      {stats['mean_sym_kl']:.4f} ± {stats['std_sym_kl']:.4f}  (n={stats['num_steps']} steps)\n"
            f"    Spearman: {sp}"
        )

    return result


def analyze_disagreement_vs_reward(json_path: str, experiment_id: int | None = None) -> dict:
    """
    For each killable mutant, collect (episode_avg_sym_kl, episode_avg_spearman, reward,
    tests_to_kill) to enable downstream correlation analysis (e.g. does higher committee
    disagreement correlate with more tests needed to find the kill?).

    Returns:
    {
        "data_points": [
            {
                "experiment_id": int,
                "run_idx": int,
                "mutant_id": str,
                "operator": str,
                "episode_avg_sym_kl": float,
                "episode_avg_spearman": float | null,
                "reward": float,
                "tests_to_kill": int
            },
            ...
        ],
        "correlation_kl_vs_tests_to_kill": float | None,   # Spearman
        "correlation_sp_vs_tests_to_kill": float | None,
    }
    """
    data = _load_json(json_path)
    experiments = data.get("experiments", [])

    if experiment_id is not None:
        experiments = [e for e in experiments if e["experiment_id"] == experiment_id]

    points = []
    for exp in experiments:
        eid = exp["experiment_id"]
        rid = exp["run_idx"]
        for mutant in exp.get("mutants", []):
            if not mutant["killable"]:
                continue
            if mutant["tests_to_kill"] is None:
                continue
            points.append(
                {
                    "experiment_id": eid,
                    "run_idx": rid,
                    "mutant_id": mutant["mutant_id"],
                    "operator": mutant["operator"],
                    "episode_avg_sym_kl": mutant["episode_avg_sym_kl"],
                    "episode_avg_spearman": mutant["episode_avg_spearman"],
                    "reward": mutant["reward"],
                    "tests_to_kill": mutant["tests_to_kill"],
                }
            )

    corr_kl = None
    corr_sp = None

    try:
        from scipy.stats import spearmanr

        kl_vals = [p["episode_avg_sym_kl"] for p in points if p["episode_avg_sym_kl"] is not None]
        ttk_vals = [p["tests_to_kill"] for p in points if p["episode_avg_sym_kl"] is not None]
        if len(kl_vals) > 2:
            c, _ = spearmanr(kl_vals, ttk_vals)
            corr_kl = float(c) if np.isfinite(c) else None

        sp_vals = [p["episode_avg_spearman"] for p in points if p["episode_avg_spearman"] is not None]
        ttk_sp = [p["tests_to_kill"] for p in points if p["episode_avg_spearman"] is not None]
        if len(sp_vals) > 2:
            c, _ = spearmanr(sp_vals, ttk_sp)
            corr_sp = float(c) if np.isfinite(c) else None
    except ImportError:
        pass

    logging.debug(f"\n--- Disagreement vs reward ({len(points)} killable mutant episodes) ---")
    logging.debug(f"  Spearman(KL, tests_to_kill):        {corr_kl}")
    logging.debug(f"  Spearman(rank_corr, tests_to_kill): {corr_sp}")

    return {
        "data_points": points,
        "correlation_kl_vs_tests_to_kill": corr_kl,
        "correlation_sp_vs_tests_to_kill": corr_sp,
    }


def analyze_agent_agreement(json_path: str, experiment_id: int | None = None) -> dict:
    """
    Analyse top-3 committee agreement across all steps in matching experiments.

    Agreement levels per step:
      full    — all agents ranked chosen_test_idx first
      soft    — chosen_test_idx in every agent's top-3 (not all #1)
      partial — chosen_test_idx in at least one agent's top-3
      none    — no agent had chosen_test_idx in their top-3

    Returns:
    {
        "total_steps": int,
        "rates": {"full": float, "soft": float, "partial": float, "none": float},
        "by_killable": {
            True:  {"total_steps": int, "rates": {...}},
            False: {"total_steps": int, "rates": {...}},
        },
        "disagreement_outcome_correlation": {
            # For killable mutants: does more 'none' disagreement mean more tests needed?
            "spearman_none_rate_vs_tests_to_kill": float | None,
            "spearman_full_rate_vs_tests_to_kill": float | None,
        },
        "per_experiment": {
            <experiment_id>: {
                "rates": {...},
                "by_run": {<run_idx>: {"rates": {...}, "total_steps": int}}
            }
        }
    }
    """
    data = _load_json(json_path)
    experiments = data.get("experiments", [])

    if experiment_id is not None:
        experiments = [e for e in experiments if e["experiment_id"] == experiment_id]

    def _empty_buckets():
        return {"full": 0, "soft": 0, "partial": 0, "none": 0}

    def _rates(buckets: dict) -> dict:
        total = sum(buckets.values())
        if total == 0:
            return {k: None for k in buckets}
        return {k: round(v / total, 4) for k, v in buckets.items()}

    global_buckets = _empty_buckets()
    killable_buckets   = {True: _empty_buckets(), False: _empty_buckets()}
    per_experiment: dict = {}

    # For correlation analysis: collect (none_rate, tests_to_kill) per killable mutant
    none_rates_killable: list[float] = []
    full_rates_killable: list[float] = []
    ttk_killable: list[int] = []

    for exp in experiments:
        eid = exp["experiment_id"]
        rid = exp["run_idx"]

        if eid not in per_experiment:
            per_experiment[eid] = {"buckets": _empty_buckets(), "by_run": {}}
        per_experiment[eid]["by_run"][rid] = {"buckets": _empty_buckets(), "total_steps": 0}

        for mutant in exp.get("mutants", []):
            killable = mutant.get("killable", False)
            ar = mutant.get("agreement_rates", {})

            for step in mutant.get("steps", []):
                lvl = step.get("agreement")
                if lvl not in global_buckets:
                    continue
                global_buckets[lvl] += 1
                killable_buckets[killable][lvl] += 1
                per_experiment[eid]["buckets"][lvl] += 1
                per_experiment[eid]["by_run"][rid]["buckets"][lvl] += 1
                per_experiment[eid]["by_run"][rid]["total_steps"] += 1

            # Collect correlation data
            if killable and mutant.get("tests_to_kill") is not None:
                if ar.get("none") is not None:
                    none_rates_killable.append(ar["none"])
                    full_rates_killable.append(ar.get("full", 0.0))
                    ttk_killable.append(mutant["tests_to_kill"])

    # Compute correlations
    corr_none = None
    corr_full = None
    try:
        from scipy.stats import spearmanr
        if len(none_rates_killable) > 2:
            c, _ = spearmanr(none_rates_killable, ttk_killable)
            corr_none = float(c) if np.isfinite(c) else None
            c, _ = spearmanr(full_rates_killable, ttk_killable)
            corr_full = float(c) if np.isfinite(c) else None
    except ImportError:
        pass

    # Build final structure
    result = {
        "total_steps": sum(global_buckets.values()),
        "rates": _rates(global_buckets),
        "by_killable": {
            "killable":     {"total_steps": sum(killable_buckets[True].values()),
                             "rates": _rates(killable_buckets[True])},
            "unkillable":   {"total_steps": sum(killable_buckets[False].values()),
                             "rates": _rates(killable_buckets[False])},
        },
        "disagreement_outcome_correlation": {
            "spearman_none_rate_vs_tests_to_kill": corr_none,
            "spearman_full_rate_vs_tests_to_kill": corr_full,
        },
        "per_experiment": {
            eid: {
                "rates": _rates(payload["buckets"]),
                "by_run": {
                    rid: {
                        "rates": _rates(run["buckets"]),
                        "total_steps": run["total_steps"],
                    }
                    for rid, run in payload["by_run"].items()
                },
            }
            for eid, payload in per_experiment.items()
        },
    }

    # Print summary
    logging.debug("\n--- Committee top-3 agreement analysis ---")
    r = result["rates"]
    logging.debug(f"  Total steps analysed: {result['total_steps']}")
    logging.debug(f"  Full agreement  (all agents #1): {r['full']:.1%}")
    logging.debug(f"  Soft agreement  (all in top-3) : {r['soft']:.1%}")
    logging.debug(f"  Partial         (≥1 in top-3)  : {r['partial']:.1%}")
    logging.debug(f"  No agreement    (none in top-3): {r['none']:.1%}")
    logging.debug(f"\n  Killable mutants only:")
    rk = result["by_killable"]["killable"]["rates"]
    if any(v is not None for v in rk.values()):
        logging.debug(f"    Full: {rk['full']:.1%}  Soft: {rk['soft']:.1%}  "
              f"Partial: {rk['partial']:.1%}  None: {rk['none']:.1%}")
    c = result["disagreement_outcome_correlation"]
    if c["spearman_none_rate_vs_tests_to_kill"] is not None:
        logging.debug(f"\n  Spearman(none_rate, tests_to_kill): "
              f"{c['spearman_none_rate_vs_tests_to_kill']:.4f}")
        logging.debug(f"  Spearman(full_rate, tests_to_kill): "
              f"{c['spearman_full_rate_vs_tests_to_kill']:.4f}")
    logging.debug("------------------------------------------\n")

    return result




def _summarise_run(exp: dict) -> dict:
    """Extract per-run summary stats (recomputed from raw mutant data for accuracy)."""
    mutants = exp.get("mutants", [])
    kl = [m["episode_avg_sym_kl"] for m in mutants if m["episode_avg_sym_kl"] is not None]
    sp = [m["episode_avg_spearman"] for m in mutants if m["episode_avg_spearman"] is not None]
    rewards = [m["reward"] for m in mutants if m["reward"] is not None]
    ttk = [m["tests_to_kill"] for m in mutants if m["tests_to_kill"] is not None]

    s = exp.get("summary") or {}
    return {
        "run_idx": exp["run_idx"],
        "mean_sym_kl": _safe_mean(kl),
        "std_sym_kl": _safe_std(kl),
        "mean_spearman": _safe_mean(sp),
        "std_spearman": _safe_std(sp),
        "mean_reward": _safe_mean(rewards),
        "std_reward": _safe_std(rewards),
        "mean_tests_to_kill": _safe_mean(ttk),
        "std_tests_to_kill": _safe_std(ttk),
        "total_tests_executed": s.get("total_tests_executed"),
        "total_tests_on_killable": s.get("total_tests_on_killable"),
        "execution_time_ms": s.get("execution_time_ms"),
        "num_killable_mutants": len(ttk),
        "num_mutants": len(mutants),
        # Agreement rates (pre-aggregated in summary block)
        "mean_agreement_rate_full":    s.get("mean_agreement_rate_full"),
        "mean_agreement_rate_soft":    s.get("mean_agreement_rate_soft"),
        "mean_agreement_rate_partial": s.get("mean_agreement_rate_partial"),
        "mean_agreement_rate_none":    s.get("mean_agreement_rate_none"),
    }


def _aggregate_run_stats(runs: list[dict]) -> dict:
    keys = ["mean_sym_kl", "mean_spearman", "mean_reward", "mean_tests_to_kill"]
    result = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        std_key = "std_" + k[len("mean_"):]
        result[f"across_runs_{k}"] = _safe_mean(vals)
        result[f"across_runs_{std_key}"] = _safe_std(vals)

    exec_times = [r["execution_time_ms"] for r in runs if r.get("execution_time_ms") is not None]
    result["mean_execution_time_ms"] = _safe_mean(exec_times)
    result["num_runs"] = len(runs)
    return result


def _print_analysis(result: dict) -> None:
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT ANALYSIS REPORT")
    logging.info("=" * 60)

    for eid, payload in result["by_experiment"].items():
        logging.info(f"\n  Experiment {eid}")
        logging.info(f"  {'─' * 40}")
        for rid, stats in payload["by_run"].items():
            logging.info(f"    Run {rid}:")
            logging.info(f"      KL divergence   : {stats['mean_sym_kl']:.4f} ± {stats['std_sym_kl']:.4f}")
            sp_str = (
                f"{stats['mean_spearman']:.4f} ± {stats['std_spearman']:.4f}"
                if stats["mean_spearman"] is not None
                else "n/a"
            )
            logging.info(f"      Rank correlation: {sp_str}")
            if stats["mean_reward"] is not None:
                logging.info(f"      Reward          : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            if stats["mean_tests_to_kill"] is not None:
                logging.info(f"      Tests to kill   : {stats['mean_tests_to_kill']:.2f} ± {stats['std_tests_to_kill']:.2f}")
            logging.info(f"      Mutants (killable/total): {stats['num_killable_mutants']}/{stats['num_mutants']}")
            if stats.get("mean_agreement_rate_full") is not None:
                logging.info(
                    f"      Agreement — full: {stats['mean_agreement_rate_full']:.1%}  "
                    f"soft: {stats['mean_agreement_rate_soft']:.1%}  "
                    f"partial: {stats['mean_agreement_rate_partial']:.1%}  "
                    f"none: {stats['mean_agreement_rate_none']:.1%}"
                )

        ar = payload["across_runs"]
        logging.info(f"    Across {ar['num_runs']} runs:")
        logging.info(f"      KL divergence   : {ar['across_runs_mean_sym_kl']:.4f} ± {ar['across_runs_std_sym_kl']:.4f}")
        sp_str = (
            f"{ar['across_runs_mean_spearman']:.4f} ± {ar['across_runs_std_spearman']:.4f}"
            if ar.get("across_runs_mean_spearman") is not None
            else "n/a"
        )
        logging.info(f"      Rank correlation: {sp_str}")

    g = result["global"]
    logging.info(f"\n  GLOBAL ({g['num_runs_total']} runs across {g['num_experiments']} experiment(s))")
    logging.info(f"  {'─' * 40}")
    logging.info(f"    KL divergence   : {g['mean_sym_kl']:.4f} ± {g['std_sym_kl']:.4f}")
    sp_str = (
        f"{g['mean_spearman']:.4f} ± {g['std_spearman']:.4f}"
        if g["mean_spearman"] is not None
        else "n/a"
    )
    logging.info(f"    Rank correlation: {sp_str}")
    if g["mean_reward"] is not None:
        logging.info(f"    Reward          : {g['mean_reward']:.2f} ± {g['std_reward']:.2f}")
    if g["mean_tests_to_kill"] is not None:
        logging.info(f"    Tests to kill   : {g['mean_tests_to_kill']:.2f} ± {g['std_tests_to_kill']:.2f}")
    logging.info("=" * 60 + "\n")