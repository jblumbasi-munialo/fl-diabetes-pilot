# FL Diabetes Pilot

A minimal federated learning platform for diabetes risk prediction using Fed-BioMed.

## Phase 1: Core FL Pipeline

This repository contains:

- `generate_data.py` – creates synthetic diabetes datasets for two simulated hospitals (node_A and node_B)
- `my_training_plan.py` – defines a simple neural network and the Fed-BioMed training plan
- `run_experiment.py` – orchestrates a federated averaging experiment

## How to Use (Locally)

1. Install Fed-BioMed: `pip install fedbiomed[researcher,node]`
2. Generate data: `python generate_data.py`
3. Start two node terminals:
   - `fedbiomed node --path ./node_A start`
   - `fedbiomed node --path ./node_B start`
4. Upload datasets to each node (interactive prompts)
5. Run the experiment: `python run_experiment.py`

## Next Steps

- Replace synthetic data with real dataset (e.g., PIMA Indians)
- Add differential privacy
- Build dashboard and API (Phases 2+)
