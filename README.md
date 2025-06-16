# FYP
## Optimisation of a multi-integrated energy system for a smart city

This project focuses on the optimisation of a multi-integrated energy system tailored for a smart city environment. It leverages real-world solar and wind data to model and simulate the operation of distributed energy resources, including photovoltaic systems, wind turbines, battery storage, and electric vehicles. Using Pyomo and Gurobi, the model evaluates cost and emission trade-offs under various scenarios. The following python scripts include the models used for the different scenarios. 

---

## 📄 Script Descriptions

| Script | Description |
|--------|-------------|
| **Scenario 1.py** | Implements the base-case optimisation for a smart city energy system, considering solar, wind, battery, EVs, and demand shifting. Focuses on generation dispatch |
| **Scenario 2.py** | Builds on Scenario 1 by incorporating V2G and BESS capabilities. |
| **Scenario 3.py** | Adds onto Scenario 2 with the feature of demand shifting. |
| **Future_scenario.py** | Similar to Scenario 3, but assumes an improved future grid (e.g. lower carbon intensity, increased renewables). Evaluates trade-offs under futuristic conditions. |
| **Future Scenario with varying shiftable load.py** | Extension of the future scenario that explores how varying the proportion of shiftable load affects system performance and optimal energy mix. |
| **Storage Capacity Sensitivty.py** | Conducts a sensitivity analysis on battery and EV storage capacities. Assesses how increasing or limiting storage affects emissions and costs. |
| **Uncertainty Scenario.py** | Introduces load uncertainty using demand scenarios. Models the system as a stochastic optimization problem with multiple demand realizations. |
| **Uncertainty Scenario 2.py** | Extends the uncertainty modeling by including uncertainty in both solar irradiance and wind speed, resulting in more robust decision-making. |
| **solve.log** | Output log file from a solver run. Useful for debugging or performance monitoring of Gurobi optimization tasks. |
