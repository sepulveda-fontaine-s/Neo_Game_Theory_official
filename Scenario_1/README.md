# Scenario 1 — Human arbitration

Scenario 1 uses `lambda=1` for Human selection and `lambda=0` for AI
selection. Agreement executes the AI proposal, disagreement executes the Human
proposal, and contextual execution selects the maximum-probability contextual
selector after mapping selector one to Human. Exact contextual ties favour the
Human.

The AI policy updates after every decision. The Human policy updates only in the
agreement region. Structural reward and the Bellman backup use the utility of the
owner of the selected proposal; EWMA tables are utility-credit diagnostics only.

## Run

```bash
python -m Scenario_1.main --H 200
```

Output:

```text
Scenario_1/Scenario1_outputs_H_200/
├── scenario1_all_runs.csv
├── scenario1_best_joint_allruns.csv
├── scenario1_best_joint_final.csv
├── scenario1_best_joint_final_H_200_Sc1.csv
└── plots/
```

`config.py` contains only operational settings. The complete model specification and
17,640-run grid are in `utils_scenario_1/grid.py`.
