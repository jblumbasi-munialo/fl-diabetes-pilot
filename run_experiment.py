from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

# When running locally, replace these with actual node IDs from your terminals
# For initial simulation without manually starting nodes, Fed-BioMed can auto-discover.
# But for the proper two-node setup, uncomment the add_break line.

exp = Experiment(
    tags=['diabetes', 'pilot'],
    training_plan_class='my_training_plan.DiabetesTrainingPlan',
    training_plan_kwargs={'model_args': {'input_dim': 8}},
    round_limit=3,
    aggregator=FedAverage(),
)

# Uncomment and add your actual node IDs when you have them:
# exp.add_break(node_ids=['node_A_uuid_here', 'node_B_uuid_here'])

exp.run()
