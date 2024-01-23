"""Differentially private algorithm to be used with FedAvg strategy."""
import logging
import random
from typing import Any, Optional

import numpy as np
import torch
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.grad_sample.utils import wrap_model
from opacus.optimizers.optimizer import DPOptimizer
from opacus.privacy_engine import PrivacyEngine
from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_fed_avg_algo import TorchFedAvgAlgo
from substrafl.remote import remote_data
from substrafl.strategies.schemas import FedAvgAveragedState, FedAvgSharedState

logger = logging.getLogger(__name__)


class TorchDPFedAvgAlgo(TorchFedAvgAlgo):
    """To be inherited.

    Wraps the necessary operation so a torch model can be trained in the Federated
    Averaging strategy using DP.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        dataset: torch.utils.data.Dataset,
        num_updates: int,
        batch_size: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        with_batch_norm_parameters: bool = False,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        num_rounds: int = None,
        *args,
        **kwargs,
    ):
        """Instantiate a TorchDPFedAvgAlgo.

        Parameters
        ----------
        model : torch.nn.modules.module.Module
            A torch model.
        criterion : torch.nn.modules.loss._Loss
            A torch criterion (loss).
        optimizer : torch.optim.Optimizer
            A torch optimizer linked to the model.
        dataset : torch.utils.data.Dataset
            Refer to the doc of the parent class.
            This behavior can be changed by re-writing the `_local_train` or
            `predict` methods.
        num_updates : int
            The number of updates to perform. Note that here we do not use
            NpIndexGenerators.
        batch_size : int
            The batch-size to target in expectation (Poisson sampling).
        scheduler : torch.optim.lr_scheduler._LRScheduler, Optional
            A torch scheduler that will be called at every batch. If None, no
            scheduler will be used. Defaults to None.
        with_batch_norm_parameters : bool
            Whether to include the batch norm layer parameters in the federated
            average strategy. Defaults to False.
        seed : typing.Optional[int]
            Seed set at the algo initialization on each organization.
            Defaults to None.
        use_gpu : bool
            Whether to use the GPUs if they are available. Defaults to True.
        dp_target_epsilon : float
            The target epsilon for (epsilon, delta)-differential private guarantee.
            Defaults to None.
        dp_target_delta : float
            The target delta for (epsilon, delta)-differential private guarantee.
            Defaults to None.
        dp_max_grad_norm : float
            The maximum L2 norm of per-sample gradients; used to enforce
            differential privacy. Defaults to None.
        num_rounds : int
            The number of rounds used to train the algo. Although this is very
            peculiar for a substra Algorithm to need access to this quantity,
            Opacus needs the number of rounds and updates used to compute the
            total number of training steps in order to compute a noise level
            respecting user constraints.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            scheduler=scheduler,
            seed=seed,
            use_gpu=use_gpu,
            index_generator=None,
            *args,
            **kwargs,
        )
        self._with_batch_norm_parameters = with_batch_norm_parameters
        self.dp_target_delta = dp_target_delta
        self.dp_target_epsilon = dp_target_epsilon
        self.dp_max_grad_norm = dp_max_grad_norm
        self.num_rounds = num_rounds

        self._apply_dp = (
            (self.dp_target_epsilon is not None)
            and (self.dp_max_grad_norm is not None)
            and (self.dp_target_delta is not None)
        )

        if not (self._apply_dp):
            raise ValueError(
                "Do not use this Algo without DP you risk running into batch"
                " sampling issues, instead use TorchFedAvgAlgo with NpIndexGenerator"
            )
        if self.num_rounds is None:
            raise ValueError(
                "if you want to perform DP-training you need to prespecify the"
                " number of rounds in advance."
            )
        self.num_updates = num_updates
        self.batch_size = batch_size

        self.num_total_steps = self.num_updates * self.num_rounds

    def _local_train(
        self,
        train_dataset: torch.utils.data.Dataset,
    ):
        """Contain the local training loop.

        Train the model on ``num_updates`` minibatches for the torch dataset.

        Parameters
        ----------
            train_dataset : torch.utils.data.Dataset
                train_dataset build from the x and y returned by the opener.
        """
        # Create torch dataloader it is important that it has a self.batch_size
        # batch size as len(train_data_loader) will be called by opacus
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        if not hasattr(self, "size_train_dataset"):
            self.size_train_dataset = len(train_dataset)

        if not hasattr(
            self, "accountant"
        ):  # if the attribute is not already there, need to instantiate the Engine
            # Important to use RDP to be able to use high epsilons
            # see https://github.com/pytorch/opacus/issues/604
            privacy_engine = PrivacyEngine(accountant="rdp")

            if not hasattr(self, "sample_rate"):
                self.sample_rate = self.batch_size / len(train_dataset)
            else:
                assert np.allclose(
                    self.sample_rate, self.batch_size / self.size_train_dataset
                ), "The length of the dataset has changed"

            # We will need it later
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.dp_target_epsilon,
                target_delta=self.dp_target_delta,
                sample_rate=self.sample_rate,
                steps=self.num_total_steps,
                accountant=privacy_engine.accountant.mechanism(),
            )

            (
                self._model,
                self._optimizer,
                train_data_loader,
            ) = privacy_engine.make_private(
                module=self._model,
                optimizer=self._optimizer,
                data_loader=train_data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
                poisson_sampling=True,
            )
            self.accountant = privacy_engine.accountant

        else:
            train_data_loader = DPDataLoader.from_data_loader(train_data_loader)

        for x_batch, y_batch in train_data_loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            # As batch-size is variable sometimes the batch is empty
            if x_batch.nelement() == 0:
                continue
            # Forward pass
            y_pred = self._model(x_batch)

            # Compute Loss
            loss = self._criterion(y_pred, y_batch)

            self._optimizer.zero_grad()
            loss.backward()

            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

    @remote_data
    def train(
        self,
        datasamples: Any,
        shared_state: Optional[FedAvgAveragedState] = None,
    ) -> FedAvgSharedState:
        """Train method of the DP federated averaging strategy.

        This method is essentially the same as the regular federated average
        algorithm but without an index generator.

        Parameters
        ----------
        datasamples : typing.Any
            Input data returned by the ``get_data`` method from the opener.
        shared_state : FedAvgAveragedState, Optional
            Dictionary containing torch parameters that will be set to the model.
            Defaults to None.

        Returns
        -------
        FedAvgSharedState
            Weight update (delta between fine-tuned weights and previous weights).
        """
        # Note that we don't simply inherit from the method from FedAvgTorchAlgo
        # because it assumes the existence of the NpIndexGenerator

        # Create torch dataset
        train_dataset = self._dataset(datasamples, is_inference=False)

        if shared_state is not None:
            # The shared states is the average of the model parameter updates
            # for all organizations
            # Hence we need to add it to the previous local state parameters
            parameter_updates = [
                torch.from_numpy(x).to(self._device)
                for x in shared_state.avg_parameters_update
            ]
            weight_manager.increment_parameters(
                model=self._model,
                updates=parameter_updates,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )

        old_parameters = weight_manager.get_parameters(
            model=self._model,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        # Train mode for torch model
        self._model.train()

        # Train the model
        self._local_train(train_dataset)

        self._model.eval()

        parameters_update = weight_manager.subtract_parameters(
            parameters=weight_manager.get_parameters(
                model=self._model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            ),
            parameters_to_subtract=old_parameters,
        )

        # Re set to the previous state
        weight_manager.set_parameters(
            model=self._model,
            parameters=old_parameters,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        return FedAvgSharedState(
            n_samples=len(train_dataset),
            parameters_update=[p.cpu().detach().numpy() for p in parameters_update],
        )

    def _local_predict(
        self,
        predict_dataset: torch.utils.data.Dataset,
        predictions_path,
        return_predictions=False,
    ):
        """Predict.

        Parameters
        ----------
        predict_dataset : torch.utils.data.Dataset
            Predict dataset built from the `x` returned by the opener.

        Important
        ---------
        The responsibility is on the user to save the computed predictions.
        Substrafl provides the `TorchAlgo._save_predictions` method for this
        purpose.
        The user can load those predictions from a metric file with the command:
        `y_pred = np.load(inputs['predictions'])`.

        Raises
        ------
        BatchSizeNotFoundError
            No default batch size has been found to perform local prediction.
            Please override the predict function of your algorithm.
        """
        # Note that we don't simply inherit from the method from FedAvgTorchAlgo
        # because it assumes the existence of the NpIndexGenerator

        predict_loader = torch.utils.data.DataLoader(
            predict_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        self._model.eval()

        predictions = []
        with torch.no_grad():
            for x in predict_loader:
                x = x.to(self._device)
                predictions.append(self._model(x))
        predictions = torch.cat(predictions, 0)
        predictions = predictions.cpu().detach()
        if return_predictions:
            return predictions
        else:
            self._save_predictions(predictions, predictions_path)

    def _get_state_to_save(self) -> dict:
        """Get all attibutes to save and pass on to next state.

        Returns
        -------
        dict
            The dict with all quantities to persist.
        """
        checkpoint = super()._get_state_to_save()

        list_attrs_to_save = [
            "dp_max_grad_norm",
            "dp_target_epsilon",
            "dp_target_delta",
            "num_rounds",
            "num_updates",
            "num_total_steps",
            "batch_size",
        ]
        list_of_attrs_after_train = [
            "noise_multiplier",
            "sample_rate",
            "size_train_dataset",
        ]
        # For some reason this method is called before ever calling train so
        # at first it doesn't have an accountant
        if hasattr(self, "accountant"):
            checkpoint["privacy_accountant_state_dict"] = self.accountant.state_dict()
            list_attrs_to_save += list_of_attrs_after_train

        for attr in list_attrs_to_save:
            checkpoint[attr] = getattr(self, attr)

        return checkpoint

    def _update_from_checkpoint(self, checkpoint: dict) -> None:
        """Set self attributes using saved values.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint to load.

        Returns
        -------
        dict
            The emptied checkpoint.
        """
        # One cannot simply call checkpoint = super()._update_from_checkpoint(chkpt)
        # because we have to change the model class if it should be changed
        # (and optimizer) aka if we find a specific key in the checkpoint

        # For some reason substrafl save and load client before calling train
        if "privacy_accountant_state_dict" in checkpoint:

            self.accountant = RDPAccountant()
            self.accountant.load_state_dict(
                checkpoint.pop("privacy_accountant_state_dict")
            )
            self.sample_rate = checkpoint.pop("sample_rate")
            self.size_train_dataset = checkpoint.pop("size_train_dataset")
            self.noise_multiplier = checkpoint.pop("noise_multiplier")
            # The init is messing up the fact that the model has become
            # a grad sampler and the optimizer a DPOptimizer, their classes
            # do not persist between serializations
            # Those lines will allow to load corresponding state_dicts wo errors
            if not isinstance(self._model, GradSampleModule):
                self._model = wrap_model(self._model, grad_sample_mode="hooks")

            if not isinstance(self._optimizer, DPOptimizer):
                self._optimizer = DPOptimizer(
                    self._optimizer,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.dp_max_grad_norm,
                    expected_batch_size=self.batch_size,
                )

            self._optimizer.attach_step_hook(
                self.accountant.get_optimizer_hook_fn(self.sample_rate)
            )

        self._model.load_state_dict(checkpoint.pop("model_state_dict"))

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint.pop("optimizer_state_dict"))

        if self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint.pop("scheduler_state_dict"))

        self._index_generator = checkpoint.pop("index_generator")

        random.setstate(checkpoint.pop("random_rng_state"))
        np.random.set_state(checkpoint.pop("numpy_rng_state"))

        if self._device == torch.device("cpu"):
            torch.set_rng_state(checkpoint.pop("torch_rng_state").to(self._device))
        else:
            torch.cuda.set_rng_state(checkpoint.pop("torch_rng_state").to("cpu"))

        attr_names = [
            "dp_max_grad_norm",
            "dp_target_epsilon",
            "dp_target_delta",
            "num_rounds",
            "num_updates",
            "num_total_steps",
            "batch_size",
        ]

        for attr in attr_names:
            setattr(self, attr, checkpoint.pop(attr))

        return
