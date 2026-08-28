# Neo-Game Theory — Part I & Part II: Scenarios 1, 2 and 3

Repository corresponding to simulations of papers: 

a) **An Entropy-Based Framework for Hybrid Coalitions in Game
Theory—Part I: Human Arbitration**

b) **An Entropy-Based Framework for Hybrid Coalitions in Game Theory—
Part II: AI Control and Human--AI Negotiation**

for the Neo-Game Theory binary Human–AI delegation framework,
designed to host the three scenarios *running independently* or as a *sequential chain*: 

```text
Scenario 1 → Scenario 2 → Scenario 3
```

The repository is being released in stages, following the publication of the
corresponding papers. The current public release contains the implementation
associated with Part I: **An Entropy-Based Framework for Hybrid Coalitions in Game
Theory—Part I: Human Arbitration**. 

Code for the Part II scenarios will be added when Part II
is published.

## Manuscripts, code, and supplementary material

This repository is intended to become the common archive for all three Neo-Game
Theory scenarios, together with the code, simulation results and supplementary material associated with
their respective papers. 
At present, only Part I, its Scenario 1 implementation, and
its supplementary material are included in the public release.

The implementations corresponding to Part II (Scenarios 2 and 3), together with
Part II and its supplementary material, will be added when Part II is published.
This staged release keeps the public repository aligned with the publication status
of the underlying research.

## Architecture

The structure below represents the complete target repository. Components associated
with Part II will become part of the public release when Part II is published.


```text
general_formulation/
├── bellman.py
├── contextual.py                 # neutral q_one / q_zero contextual probabilities
├── csv_outputs.py                # CSV schema enforcement and deterministic writing utilities
├── entropy.py                    # public and fast binary Shannon entropy and Jensen--Shannon divergence
├── frequencies.py                # cumulative execution, regime, contextual, and state frequencies
├── grid_common.py                # common grid preparation only
├── identifiers.py                # stable identifiers and artifact filenames for simulation outputs
├── inheritance.py                # self-contained chain checkpoint payload
├── learning_rates.py             # three eta schedules
├── numerics.py                   # probability and simplex utilities: validation, uniform and one-hot construction, and numerical-tolerance handling.
├── plots.py                      # shared plotting primitives for the four PNG artifacts produced by all scenarios
├── policy_updates.py             # fast binary updates
├── state_generation.py           # cached kernels and binary sampling
├── utility_credit.py             # owner-specific EWMA utility-credit traces.
└── validation.py                 # general mathematical, temporal, probability, and output validations.


Scenario_1
├── README_Scenario_1.md
├── supplementary/
│   ├── Supplementary_material_Sc1_Ch_I.pdf
│   └── Supplementary_material_Sc1_Ch_II.pdf
├── __init__.py                   # package marker
├── config.py                     # operational settings for Scenario 1.
├── main.py                       # end-to-end orchestration for Scenario 1
└── utils_scenario_1/
      ├── __init__.py             # Scenario 1 Human-arbitration implementation modules.
      ├── delegation.py           # Scenario 1 delegation under Human arbitration
      ├── execution.py            # Scenario 1 proposal sampling and executed-action resolution.
      ├── grid.py                 # complete Scenario 1 grid and model specification
      ├── output_schema.py        # ordered Scenario 1 CSV output contracts.
      ├── selection.py            # Scenario 1 run ranking and representative-run selection.
      ├── simulation.py           # fast Scenario 1 decision loop with boundary-only validation.
      ├── simulation_reference.py # Scenario 1 decision loop implementing the temporal contract.
      ├── update_rules.py          # Scenario 1 asymmetric policy-update eligibility and execution.


Scenario_2
├── README_Scenario_2.md
├── __init__.py                   # Scenario 2: AI-control simulation package.
├── config.py                     # operational settings for Scenario 2.
├── main.py                       # end-to-end orchestration for Scenario 2.
└──utils_scenario_2/
      ├── __init__.py             # Scenario 2 AI-control implementation modules.
      ├── delegation.py           # Scenario 2 delegation under AI control.
      ├── execution.py            # Scenario 2 proposal sampling and execution resolution.
      ├── grid.py                 # independent and inherited Scenario 2 grids
      ├── output_schema.py        # ordered CSV contracts for Scenario 2.
      ├── selection.py            # Scenario 2 run ranking and representative-run selection.
      ├── simulation.py           # fast Scenario 2 production kernel with boundary-only validation.
      ├── update_rules.py          # Scenario 2 policy-update eligibility and execution.


Scenario_3/
├── README_Scenario_3.md
├── __init__.py                   # Scenario 3: Human--AI negotiation simulation package.
├── config.py                     # operational settings for Scenario 3.
├── main.py                       # end-to-end orchestration for Scenario 3.
├── utils_scenario_3/
│   ├── __init__.py               # Scenario 3 negotiation implementation modules.
│   ├── delegation.py             # Scenario 3 delegation, agreement, and finite-round negotiation.
│   ├── execution.py              # Scenario 3 proposal sampling and executed-action resolution.
│   ├── grid.py                   # independent and inherited Scenario 3 grids.
│   ├── output_schema.py          # ordered CSV contracts for Scenario 3.
│   ├── selection.py              # Scenario 3 run ranking and representative-run selection.
│   ├── simulation.py             # fast Scenario 3 production kernel with consensus-gated Human learning.
│   └── update_rules.py           # Scenario 3 consensus-gated Human policy updates.
└── Scenario_3_parallel_main_package/
    ├── Scenario_3_parallel_main.py           # prepared parallel-grid replacement main for isolated Scenario_3_parallel runs.
    ├── SC3_PARALLEL_INSTALL.txt              # installation and execution instructions for the parallel package.
    └── SC3run_H_large_parallel_array.sbatch  # SLURM array launcher for large parallel Scenario 3 runs.

```

`config.py` in each scenario contains operational settings only: active horizon,
output filenames, plotting resolution, progress interval, and branch default. Common
scientific grid components are defined in `general_formulation/grid_common.py` and are
materialized or extended by the corresponding scenario-local `grid.py`; Scenario-3
specific extensions live in `Scenario_3/utils_scenario_3/grid.py`.

## Installation and package discovery

The experiments can be run directly from the repository root, as shown below. For
local development, an editable installation is also supported and keeps the scenario
modules linked to this checkout:

```bash
python -m pip install -e .
```

A standard local installation is likewise supported:

```bash
python -m pip install .
```

`pyproject.toml` explicitly discovers the production packages `Scenario_1*`,
`Scenario_2*`, `Scenario_3*`, and `general_formulation*`, while excluding `tests*`
from the installed distribution. This package-discovery configuration affects only
installation and import resolution; it does not modify scientific grids, seeds,
delegation rules, policy updates, or numerical simulation results. Runtime
dependencies are Python 3.10 or newer, NumPy, and Matplotlib. Pytest is optional and
can be installed together with the repository for verification:

```bash
python -m pip install -e ".[test]"
```

For the reported experiments, running from the repository root or using the editable
installation is recommended because the scenario entry points write their default
output directories beside the corresponding scenario package.

## Contextual rule

The common calculation is selector-neutral:

```python
q_one = (D_JS_T - alpha_agree) / (alpha_disagree - alpha_agree)
q_zero = 1.0 - q_one
```

Each scenario maps the selector values according to its own convention:

```text
Scenario 1: lambda=1 Human, lambda=0 AI
Scenario 2: lambda=1 AI,    lambda=0 Human
Scenario 3: lambda=1 Human, lambda=0 AI
```

After contextual entry, the selector value with maximum support is selected
deterministically. Exact ties use the scenario-specific convention. There is no
contextual random draw. Structural validation occurs when configurations or
external winner CSV files enter the system.

## Independent execution

Run from the repository root:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m Scenario_1.main --H 5000
python -m Scenario_2.main --H 5000 --branch independent-grid
python -m Scenario_3.main --H 5000 --branch independent-grid

# Substitute any horizon, e.g.
python -m Scenario_1.main --H 200
python -m Scenario_2.main --H 200 --branch independent-grid
```

## Sequential inherited chain

```bash
python -m Scenario_2.main --H 5000 --branch inherited-chain \
  --input-csv Scenario_1/Scenario1_outputs_H_5000/scenario1_best_joint_final.csv

python -m Scenario_3.main --H 5000 --branch inherited-chain \
  --input-csv Scenario_2/Scenario2_outputs_H_5000_inherited-chain/scenario2_best_joint_final.csv
```

Every `*_best_joint_final.csv` row is a self-contained checkpoint for the next
scenario. In addition to thresholds, learning schedule, beta, gamma, state mechanism,
seed and provenance, it stores:

- both terminal policies over both states and actions;
- both structural-utility tables;
- the complete resolved binary Virtual-Nature kernel.

The inherited readers reconstruct these objects only from the selected winner row.
They do not silently replace them with values from the next scenario's `config.py` or
independent grid.

## Outputs

Scenario 1:

```text
Scenario_1/Scenario1_outputs_H_<H>/
```

Scenarios 2 and 3:

```text
Scenario_2/Scenario2_outputs_H_<H>_<branch>/
Scenario_3/Scenario3_outputs_H_<H>_<branch>/
```

Each output directory contains:

```text
scenarioX_all_runs.csv
scenarioX_best_joint_allruns.csv
scenarioX_best_joint_final_H_<H>_ScX.csv
plots/
```

Each plot has been renamed in order to fit in this repository, example:

1. **Original name:**  
   <sub>scenario1__H-15000__<wbr>eta-constant__<wbr>eta0-0p05__<wbr>c-NA__<wbr>random_states-true__<wbr>pA-0p9__<wbr>alpha_agree-0p2__<wbr>alpha_disagree-0p51__<wbr>beta-0p1__<wbr>gamma-0p1__<wbr>seed-42__<wbr>id-3ed50b2768__<wbr>plot1_a1.png</sub>

   **Renamed to:**  
   <sub>sc1_15000__<wbr>eta-const-0p05__<wbr>c-NA__<wbr>rd_sts-T__<wbr>pA-0p9__<wbr>alpha_ag-0p2__<wbr>alpha_dis-0p51__<wbr>bet-0p1__<wbr>gam-0p1__<wbr>sd-42__<wbr>id-3ed50b2768__<wbr>plot1_a1.png</sub>

2. **Original name:**  
   <sub>scenario1__H-30000__<wbr>eta-exact_empirical__<wbr>eta0-NA__<wbr>c-NA__<wbr>random_states-false__<wbr>pA-NA__<wbr>alpha_agree-0p2__<wbr>alpha_disagree-0p35__<wbr>beta-0p05__<wbr>gamma-0p5__<wbr>seed-42__<wbr>id-1e7f298447__<wbr>plot1_a1.png</sub>

   **Renamed to:**  
   <sub>sc1_30000__<wbr>eta_emp-NA__<wbr>c-NA__<wbr>rd_sts-F__<wbr>pA-NA__<wbr>alpha_ag-0p2__<wbr>alpha_dis-0p35__<wbr>bet-0p05__<wbr>gam-0p5__<wbr>sd-42__<wbr>id-1e7f298447__<wbr>plot1_a1.png</sub>

3. **Original name:**  
   <sub>scenario1__H-50000__<wbr>eta-global_decay__<wbr>eta0-0p1__<wbr>c-0p002__<wbr>random_states-true__<wbr>pA-0p3__<wbr>alpha_agree-0p49__<wbr>alpha_disagree-0p85__<wbr>beta-0p1__<wbr>gamma-0p5__<wbr>seed-42__<wbr>id-b51159e288__<wbr>plot1_a1.png</sub>

   **Renamed to:**  
   <sub>sc1_50000__<wbr>eta_decay-0p1__<wbr>c-0p002__<wbr>rd_sts-T__<wbr>pA-0p3__<wbr>alpha_ag-0p49__<wbr>alpha_dis-0p85__<wbr>bet-0p1__<wbr>gam-0p5__<wbr>sd-42__<wbr>id-b51159e288__<wbr>plot1_a1.png</sub>

The `scenarioX_best_joint_final_H_<H>_ScX.csv` file carries
the horizon and scenario token in the filename. With the current single seed,
each horizon produces six winner groups (`3 eta kinds × 2 state-generation
mechanisms`) and four plots per winner.

Heavy csv files (<25M) have been converted to *.zip

## Scenario 1 implementation choices

Scenario 1 implements the Human-arbitration regime with the binary convention

```text
lambda = 1  -> Human proposal
lambda = 0  -> AI proposal
```
The scenario-specific execution rules are implemented in Scenario_1/utils_scenario_1/delegation.py:
  agreement        -> AI proposal
  contextual       -> deterministic maximum-probability proposal selection
  contextual tie   -> Human proposal
  disagreement     -> Human proposal

The contextual rule uses the common normalized selector probabilities defined in
`general_formulation/contextual.py`; no additional random draw is performed after
entry into the contextual region.

Policy-update eligibility is implemented in `Scenario_1/utils_scenario_1/update_rules.py`. 
The AI policy updates toward the executed action after every decision, whereas the Human 
policy updates only in the agreement region.

Structural reward and the Bellman backup use the structural utility associated with
the owner of the selected proposal. EWMA utility tables are diagnostic credit traces
and do not determine the Bellman reward.  See `Scenario_1/README_Scenario_1.md` for 
the complete Scenario-1 execution rules.

## Scenario 2 implementation choices

Scenario 2 implements the AI-control regime with the binary convention

```text
lambda = 1  -> AI proposal
lambda = 0  -> Human proposal
```
The scenario-specific execution rules are implemented in `Scenario_2/utils_scenario_2/delegation.py`:
 agreement        -> AI proposal
 contextual       -> deterministic maximum-probability proposal selection
 contextual tie   -> AI proposal
 disagreement     -> proposal generated by the lower-entropy policy
 entropy tie      -> AI proposal

The contextual rule uses the common normalized selector probabilities defined in
`general_formulation/contextual.py`; no additional random draw is performed after
entry into the contextual region.

Policy-update eligibility is implemented in `Scenario_2/utils_scenario_2/update_rules.py`. 
The AI policy updates toward the executed action after every decision, whereas the Human 
policy updates only in the agreement region. Effective update counts are incremented only 
when the corresponding policy update is actually applied.

Scenario 2 therefore resolves each delegation region internally. Agreement does not
invoke the entropy rule, disagreement does not redirect to contextual execution, and
the entropy comparison is evaluated only from the predecision Human and AI policies.

 See `Scenario_2/README_Scenario_2.md` for the complete Scenario-2 execution rules.

## Scenario 3 implementation choices

The Scenario-3-specific operational settings used in the reported simulations are
explicit grid fields in `Scenario_3/utils_scenario_3/grid.py`:

```text
terminal_agreement_rule = ownership-count
objective_dominance_rule = strict-pareto
N_max_negotiation = 1
informed_human_rule = accept-recommendation
```

These settings instantiate the computational specification described in the manuscript.
`ownership-count` resolves the terminal agreement branch when neither proposal is
acceptable; `strict-pareto` defines the objective-dominance check after failed
disagreement-region negotiation; `N_max_negotiation = 1` is the default negotiation
horizon used in the reported simulations; and `accept-recommendation` operationalizes
the informed-Human terminal response for simulation. In deployment, the informed-Human
response is intended to be supplied by the Human after receiving the corresponding
AI-provided information.

Agreement and disagreement resolve internally and never redirect to the contextual
selector. See `Scenario_3/README_Scenario_3.md` for the complete Scenario-3 execution rules.

## Verification

```bash
python verify_install.py
python -m pytest -q
```
The supplied tests cover the common formulation, architecture contracts,
contextual delegation and inheritance across the three scenarios, with additional
scenario-specific tests.

## Excluded

The package contains no `docs/`, `legacy/`, generated production outputs,
`__pycache__`, or `tail`-window logic.


**Code in this repository is released under the MIT License.**
