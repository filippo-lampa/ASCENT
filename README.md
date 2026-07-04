# :balance_scale: ASCENT-C: Committee-Based Neural-MCTS Test Prioritization

Multi-agent version of ASCENT for mutation-based test prioritization. The prioritization problem, mutant format, test discovery logic, and execution protocol remain ASCENT-compatible. The decision engine is changed from a single Neural-MCTS agent to a committee of specialized agents:

- **Exploitation agent**: favors historically rewarding tests.
- **Exploration agent**: favors under-visited regions of the search space.
- **Diversity agent**: favors globally under-used tests through an inverse-frequency bonus.

Each agent maintains a private policy network and private search tree. By default, the agents share one value network because all agents observe the same executed trajectory and solve the same state-value regression problem. Use `--separate_value_networks` only for ablation experiments.

## Mutant Generation

The multi-agent branch uses the same Sumo/Hardhat mutant generation workflow as the main ASCENT branch. You can skip this section if you use mutants already available in this repository.

### Prerequisites

- Node.js v14 or higher
- Python 3
- Git LFS, if downloading pre-generated mutation artifacts from the repository

### Configure Sumo in the SUT

Add the Sumo package to the SUT `package.json`:

```json
"dependencies": {
  "@morenabarboni/sumo": "file:../../morenabarboni-sumo-x.x.x.tgz"
}
```

Add reporting/parsing dependencies:

```json
"dependencies": {
  "mochawesome": "7.1.3",
  "@babel/parser": "7.25.3",
  "recast": "0.23.9",
  "solidity-coverage": "0.7.22"
}
```

Configure Mochawesome in `hardhat.config.js`:

```js
const config = {
  solidity: {
    version: "0.8.16",
    settings: {
      optimizer: {
        enabled: true,
        runs: 1000,
      },
    },
  },
  mocha: {
    reporter: "mochawesome",
    reporterOptions: {
      reportFilename: "mochawesome",
      quiet: false,
      json: true,
      html: false,
    },
  },
};
```

Install dependencies and compile:

```bash
npm install
npx hardhat compile
```

Install Surya and generate the CFG:

```bash
npm install -g surya
surya graph contracts/**/*.sol | dot > cfg.dot
```

Run Sumo and generate the coverage matrix:

```bash
npx sumo test
npx hardhat coverage --matrix
npx sumo addMutationsContext
```

## Running Multi-Agent Prioritization

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using pre-generated mutants stored with Git LFS:

```bash
git lfs fetch --all
git lfs pull
```

Run one project with default paths:

```bash
python3 prioritizer.py \
  --sut_name thorwallet \
  --aggregation_strategy weighted_mean \
  --num_runs 5
```

Default paths are resolved as:

```text
case_studies/<sut_name>/test
sumo_results/<sut_name>/mutations.json
```

Run with explicit paths:

```bash
python3 prioritizer.py \
  --sut_name thorwallet \
  --tests_folder case_studies/thorwallet/test \
  --mutants sumo_results/thorwallet/mutations.json \
  --aggregation_strategy weighted_mean \
  --num_runs 5
```

Run all aggregation strategies manually:

```bash
python3 prioritizer.py --sut_name nextgeneration --aggregation_strategy arithmetic_mean --num_runs 5
python3 prioritizer.py --sut_name nextgeneration --aggregation_strategy geometric_mean --num_runs 5
python3 prioritizer.py --sut_name nextgeneration --aggregation_strategy weighted_mean --num_runs 5
python3 prioritizer.py --sut_name nextgeneration --aggregation_strategy borda_count --num_runs 5
```

Each run writes outputs to:

```text
experiments/<sut_name>_<aggregation_strategy>_<timestamp>/
```

The output directory contains the disagreement tracker JSON, summary files, and plots.

## Command-Line Parameters

| Argument | Default | Description |
|---|---:|---|
| `--sut_name` | required | Name of the target SUT/project. |
| `--tests_folder` | `case_studies/<sut_name>/test` | Folder containing `.js`/`.ts` tests. |
| `--mutants` | `sumo_results/<sut_name>/mutations.json` | Path to the Sumo mutations file. |
| `--coverage` | unused | Accepted for compatibility with the main ASCENT CLI; this runner does not use it. |
| `--aggregation_strategy` | `weighted_mean` | Committee aggregation: `arithmetic_mean`, `geometric_mean`, `weighted_mean`, or `borda_count`. |
| `--num_runs` | `5` | Number of independent repetitions. |
| `--results_root` | `experiments` | Root directory for experiment outputs. |
| `--timestamp` | current time | Optional run suffix for reproducible output paths. |
| `--skip_analysis` | `False` | Skip post-run committee analysis. |
| `--shuffle_mutants` | `False` | Shuffle mutants independently before each run. |
| `--random_seed` | `None` | Seed used when `--shuffle_mutants` is enabled. |
| `--plot_delta` | `30` | Plot every `n` mutants when intermediate plotting is enabled. |
| `--average_delta` | `10` | Moving-average window used in plots. |
| `--buffer_size` | `#mutants` | Replay buffer size for policy/value updates. |
| `--batch_size` | `40` | Batch size for policy/value updates. |
| `--update_delta` | `1` | Update networks every `n` episodes. |
| `--rollout_delay` | `45` | Number of warm-up episodes before neural priors are used. |
| `--asymmetric_loss_alpha` | `6.0` | Underestimation penalty for the asymmetric value loss. |
| `--exploration_c_parameter` | `3.0` | PUCT exploration coefficient for the exploration agent. |
| `--exploitation_c_parameter` | `0.5` | PUCT exploration coefficient for the exploitation agent. |
| `--diversity_c_parameter` | `2.0` | PUCT exploration coefficient for the diversity agent. |
| `--diversity_bonus_weight` | `1.0` | Weight of the inverse-frequency diversity bonus. |
| `--value_network_learning_rate` | `0.001` | Learning rate for the value network. |
| `--policy_network_learning_rate` | `0.0001` | Learning rate for each policy network. |
| `--separate_value_networks` | `False` | Use one value network per agent instead of the default shared value network. |

## Examples

Weighted aggregation on THORWallet:

```bash
python3 prioritizer.py --sut_name thorwallet --aggregation_strategy weighted_mean --num_runs 5
```

Geometric aggregation on Quadrata with larger batches and less frequent updates:

```bash
python3 prioritizer.py \
  --sut_name quadrata \
  --aggregation_strategy geometric_mean \
  --batch_size 80 \
  --update_delta 5 \
  --num_runs 5
```

Ablation with separate value networks:

```bash
python3 prioritizer.py \
  --sut_name nextgeneration \
  --aggregation_strategy weighted_mean \
  --separate_value_networks \
  --num_runs 5
```

## Plotting Note

If plots do not show or save correctly, configure the Matplotlib backend for your OS. See the Matplotlib backend documentation: https://matplotlib.org/stable/users/explain/figure/backends.html

## Supplementary Material

The experimental artifacts used for the analysis presented in the paper are available on OSF at https://osf.io/jaqyc/overview?view_only=6806c30fc4cf4aff903b2eb9e1a5c1e0.
