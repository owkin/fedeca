"""Trace the graph of the compute plans to create figures and tables."""
from fedeca.utils.survival_utils import CoxData
from fedeca.strategies.fed_kaplan import FedKaplan
from fedeca.strategies.webdisco import WebDisco
from fedeca.strategies.fed_smd import FedSMD
from fedeca.fedeca_core import LogisticRegressionTorch
from substrafl.experiment import execute_experiment
from fedeca.utils.data_utils import split_dataframe_across_clients
from substrafl.nodes import AggregationNode
import os
import torch
from fedeca.utils.survival_utils import (
    CoxPHModelTorch,
)
from fedeca import FedECA
from fedeca.algorithms.torch_newton_raphson_algo_decorated import TorchNewtonRaphsonAlgoDecorated as TorchNewtonRaphsonAlgo

from substrafl.strategies import FedAvg
from fedeca.strategies.newton_raphson_decorated import NewtonRaphsonDecorated as NewtonRaphson

from torch import nn
from torch.optim import SGD

from fedeca.algorithms import TorchWebDiscoAlgo
from fedeca.algorithms.torch_dp_fed_avg_algo import TorchDPFedAvgAlgo
from fedeca.analytics import RobustCoxVariance, RobustCoxVarianceAlgo

from fedeca.utils import (
    make_substrafl_torch_dataset_class,
)
import sys

# Global FL setup for each strategy
NDIM = 10
data = CoxData(seed=42, n_samples=1000, ndim=NDIM)
df = data.generate_dataframe()

df = df.drop(columns=["propensity_scores"], axis=1)

propensity_model = LogisticRegressionTorch(ndim=NDIM)
os.makedirs("./temp", exist_ok=True)
os.makedirs("./tmp", exist_ok=True)
clients, train_data_nodes, _, _, _ = split_dataframe_across_clients(
    df,
    n_clients=4,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "treatment"},
    data_path="./temp/data",
    backend_type="simu",
)
ds_client = clients[train_data_nodes[0].organization_id]
kwargs_agg_node = {"organization_id": train_data_nodes[0].organization_id}
aggregation_node = AggregationNode(**kwargs_agg_node)



os.system("rm /Users/jterrail/Desktop/workflow.txt")

with open("/Users/jterrail/Desktop/workflow.txt", "w") as f:
    f.write("<bloc>\n<name>FedECA: propensity model training</name>\n")


logreg_dataset_class = make_substrafl_torch_dataset_class(
    ["treatment"],
    "event",
    "time",
    fit_cols=None,
    dtype="float64",
    return_torch_tensors=True,
    client_identifier=None,
)
logreg_model = propensity_model
seed = 42
l2_coeff_nr = 0.001

class NRAlgo(TorchNewtonRaphsonAlgo):
    def __init__(self):
        super().__init__(
            model=logreg_model,
            batch_size=sys.maxsize,
            criterion=nn.BCELoss(),
            dataset=logreg_dataset_class,
            seed=seed,
            l2_coeff=l2_coeff_nr,
        )

nr_algo = NRAlgo()

nr_strategy = NewtonRaphson(
    damping_factor=l2_coeff_nr,
    algo=nr_algo,
    metric_functions={},
)
compute_plan = execute_experiment(
    client=ds_client,
    strategy=nr_strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=3,
    experiment_folder="./tmp/experiment_summaries_newton_raphson",
    )
with open("/Users/jterrail/Desktop/workflow.txt", "a") as f:
    f.write("</bloc>\n")

os.system("python clean_log_file.py")
os.system("python create_tree.py")
os.system("python create_graphs.py")
os.system("cp -r /Users/jterrail/Desktop/outputs/entire_workflow_rank_0/graphs /Users/jterrail/Desktop/propensity_graphs")  # noqa: E501
os.system("cp -r /Users/jterrail/Desktop/outputs/entire_workflow_rank_0/tables /Users/jterrail/Desktop/propensity_tables")  # noqa: E501
os.system("rm -r /Users/jterrail/Desktop/outputs")
os.system("rm /Users/jterrail/Desktop/workflow.txt")
breakpoint()

with open("/Users/jterrail/Desktop/workflow.txt", "w") as f:
    f.write("<bloc>\n<name>FedECA-DP: propensity model training</name>\n")

dp_propensity_model_optimizer = SGD(
    params=logreg_model.parameters(),
    lr=0.001,
)
class DPLogRegAlgo(TorchDPFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=logreg_model,
            criterion=nn.BCELoss(),
            optimizer=dp_propensity_model_optimizer,
            dataset=logreg_dataset_class,
            seed=seed,
            num_updates=100,
            batch_size=100,
            num_rounds=3,
            dp_target_epsilon=50.,
            dp_target_delta=1e-2,
            dp_max_grad_norm=1.,
        )

dp_algo = DPLogRegAlgo()
dp_strategy = FedAvg(
    algo=dp_algo, metric_functions={},
)
# compute_plan = execute_experiment(
#     client=ds_client,
#     strategy=dp_strategy,
#     train_data_nodes=train_data_nodes,
#     evaluation_strategy=None,
#     aggregation_node=aggregation_node,
#     num_rounds=3,
#     experiment_folder="./tmp/experiment_summaries_dp_fedavg",
#     )

cox_model = CoxPHModelTorch(
    ndim=1,
    torch_dtype=torch.float64,
)
survival_dataset_class = make_substrafl_torch_dataset_class(
    ["time", "event"],
    "event",
    "time",
    fit_cols=None,
    dtype="float64",
    client_identifier=None,
)

# no self attributes in this class !!!!!!
class WDAlgo(TorchWebDiscoAlgo):
    def __init__(self, propensity_model, robust):
        super().__init__(
            model=cox_model,
            # TODO make this batch-size argument disappear from
            # webdisco algo
            batch_size=sys.maxsize,
            dataset=survival_dataset_class,
            seed=seed,
            duration_col="time",
            event_col="event",
            treated_col="treatment",
            standardize_data=True,
            penalizer=0.,
            l1_ratio=0.,
            initial_step_size=1.,
            learning_rate_strategy="lifelines",
            store_hessian=True,
            propensity_model=propensity_model,
            training_strategy="iptw",
            cox_fit_cols=None,
            propensity_fit_cols=None,
            robust=robust,
        )


webdisco_algo = WDAlgo(propensity_model=propensity_model, robust=False)
webdisco_strategy = WebDisco(
    algo=webdisco_algo,
    standardize_data=True,
    metric_functions={},
)
# compute_plan = execute_experiment(
#     client=ds_client,
#     strategy=webdisco_strategy,
#     train_data_nodes=train_data_nodes,
#     evaluation_strategy=None,
#     aggregation_node=aggregation_node,
#     num_rounds=3,
#     experiment_folder="./tmp/experiment_summaries_webdisco",
#     )

kaplan_strategy = FedKaplan(
    treated_col="treatment",
    duration_col="time",
    event_col="event",
    propensity_model=propensity_model,
    client_identifier="center",
)

# compute_plan = execute_experiment(
#     client=ds_client,
#     strategy=kaplan_strategy,
#     train_data_nodes=train_data_nodes,
#     evaluation_strategy=None,
#     aggregation_node=aggregation_node,
#     num_rounds=1,
#     experiment_folder="./tmp/experiment_summaries_fed_kaplan",
#     )

smd_strategy = FedSMD(
    treated_col="treatment",
    duration_col="time",
    event_col="event",
    propensity_model=propensity_model,
    client_identifier="center",
    use_unweighted_variance=True,
)

# compute_plan = execute_experiment(
#     client=ds_client,
#     strategy=smd_strategy,
#     train_data_nodes=train_data_nodes,
#     evaluation_strategy=None,
#     aggregation_node=aggregation_node,
#     num_rounds=1,
#     experiment_folder="./tmp/experiment_summaries_fed_smd",
#     )

# TODO A bit ugly remove call to FedECA
fed_iptw = FedECA(ndim=10, treated_col="treatment", event_col="event", duration_col="time", num_rounds_list=[2, 3], variance_method="robust")  # noqa: E501
fed_iptw.fit(df, n_clients=4, split_method="split_control_over_centers", split_method_kwargs={"treatment_info": "treatment"}, data_path="./data", backend_type="simu")  # noqa: E501

beta = fed_iptw.final_params_list[0]
variance_matrix = fed_iptw.variance_matrix
global_robust_statistics = fed_iptw.global_robust_statistics
propensity_model = fed_iptw.propensity_model
duration_col = fed_iptw.duration_col
event_col = fed_iptw.event_col
treated_col = fed_iptw.treated_col


class MyRobustCoxVarianceAlgo(RobustCoxVarianceAlgo):
    def __init__(self, **kwargs):
        super().__init__(
            beta=beta,
            variance_matrix=variance_matrix,
            global_robust_statistics=global_robust_statistics,
            propensity_model=propensity_model,
            duration_col=duration_col,
            event_col=event_col,
            treated_col=treated_col,
        )

my_robust_cox_algo = MyRobustCoxVarianceAlgo()
robust_strategy = RobustCoxVariance(algo=my_robust_cox_algo, metric_functions={})


# with open("/Users/jterrail/Desktop/workflow.txt", "a") as f:
#     f.write("</bloc>\n")

# os.system("python clean_log_file.py")
# os.system("python create_tree.py")
# os.system("python create_graphs.py")
# os.system("cp -r /Users/jterrail/Desktop/outputs/entire_workflow_rank_0/graphs /Users/jterrail/Desktop/cox_graphs")  # noqa: E501
# os.system("cp -r /Users/jterrail/Desktop/outputs/entire_workflow_rank_0/tables /Users/jterrail/Desktop/cox_tables")  # noqa: E501
# os.system("rm -r /Users/jterrail/Desktop/outputs")
# os.system("rm /Users/jterrail/Desktop/workflow.txt")



