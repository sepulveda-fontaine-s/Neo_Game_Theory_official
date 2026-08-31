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


# Plots 

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
