'''
Federated solution based on:
https://github.com/drivendataorg/pets-prize-challenge-runtime/blob/main/examples_src/pandemic/solution_federated.py

More information about the data format:
https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/643/
'''
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters,
                         Scalar)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from go import go_start, path_str
from loguru import logger

GO_EXEC = 'solution'

# LOOKAHEAD = 7


# def to_parameters_ndarrays(numerator: float, denominator: float) -> List[np.ndarray]:
#     """Utility function to convert SirModel parameters to List[np.ndarray] used by
#     Flower's NumPyClient for transferring model parameters.
#     """
#     return [np.array([numerator, denominator])]


# def from_parameters_ndarrays(parameters: List[np.ndarray]) -> Tuple[float, float]:
#     """Utility function to convert SirModel parameters from List[np.ndarray] used by
#     Flower's NumPyClient for transferring model parameters.
#     """
#     numerator, denominator = parameters[0]
#     return numerator, denominator


# def get_model_parameters(model: SirModel) -> List[np.ndarray]:
#     """Gets the paramters of a SirModel model as List[np.ndarray] used by Flower's
#     NumPyClient for transferring model parameters.
#     """
#     return to_parameters_ndarrays(model.numerator, model.denominator)


# def set_model_parameters(model: SirModel, parameters: List[np.ndarray]) -> SirModel:
#     """Sets the parameters of a SirModel model from a List[np.ndarray] used by Flower's
#     NumPyClient for transferring model parameters."""
#     numerator, denominator = from_parameters_ndarrays(parameters)
#     model.set_params(numerator=numerator, denominator=denominator)
#     return model


def train_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    return TrainClient(
      cid=cid,
      person_data_path=person_data_path,
      household_data_path=household_data_path,
      residence_location_data_path=residence_location_data_path,
      activity_location_data_path=activity_location_data_path,
      activity_location_assignment_data_path=activity_location_assignment_data_path,
      population_network_data_path=population_network_data_path,
      disease_outcome_data_path=disease_outcome_data_path,
      client_dir=client_dir,
    )


@dataclass
class TrainClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""
    cid: str
    person_data_path: Path
    household_data_path: Path
    residence_location_data_path: Path
    activity_location_data_path: Path
    activity_location_assignment_data_path: Path
    population_network_data_path: Path
    disease_outcome_data_path: Path
    client_dir: Path

    def __post_init__(self):
        """ Start Go client process for processing later """
        self.num_examples = pd.read_csv(self.disease_outcome_data_path).shape[0]
        self.client = go_start(self.client_dir / GO_EXEC, 'federated-train-client',
            '-person-data-path', path_str(self.person_data_path),
            '-household-data-path', path_str(self.household_data_path),
            '-residence-location-data-path', path_str(self.residence_location_data_path),
            '-activity-location-data-path', path_str(self.activity_location_data_path),
            '-activity-location-assignment-data-path', path_str(self.activity_location_assignment_data_path),
            '-population-network-data-path', path_str(self.population_network_data_path),
            '-disease-outcome-data-path', path_str(self.disease_outcome_data_path),
            '-client-dir', path_str(self.client_dir),
        )

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        # TODO provide real implementation

        # dummy implementation:
        # """Fit model on partitioned dataset. Server is not passing any meaningful
        # parameters or configuration. Returned fitted model parameters back to server."""
        # self.model.fit(self.disease_outcome_df)
        # return get_model_parameters(self.model), self.disease_outcome_df.shape[0], {}
        return [], self.num_examples, {}


def train_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    training_strategy = TrainStrategy(server_dir=server_dir)
    num_rounds = 1
    return training_strategy, num_rounds


@dataclass
class TrainStrategy(fl.server.strategy.Strategy):
    """Federated aggregation equivalent to pooling observations across partitions."""
    server_dir: Path

    def __post_init__(self):
        """ Start Go strategy process for processing later """
        self.strategy = go_start(self.server_dir / GO_EXEC, 'federated-train-strategy',
            '-server-dir', path_str(self.server_dir),
        )


    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Initialize the (global) model parameters.

        Parameters
        ----------
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        # TODO replace with real implementation
        return fl.common.ndarrays_to_parameters([])


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        logger.info(f"Configuring fit for round {server_round}...")

        # TODO provide real implementation

        # # Fit all clients.
        # logger.info(f"Configuring fit for round {server_round}...")
        # # Fit every client once. Don't need to pass any initial parameters or config.
        # clients = list(client_manager.all().values())
        # empty_fit_ins = fl.common.FitIns(fl.common.ndarrays_to_parameters([]), {})
        # logger.info(f"...done configuring fit for round {server_round}")
        # return [(client, empty_fit_ins) for client in clients]
        return []


    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        """Aggregate training results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        if len(failures) > 0:
            raise Exception(f"Client fit round had {len(failures)} failures.")

        # TODO provide real implementation

        # # results is List[Tuple[ClientProxy, FitRes]]
        # # convert FitRes to List[np.ndarray]
        # client_parameters_list_of_ndarrays = [
        #     fl.common.parameters_to_ndarrays(fit_res.parameters)
        #     for _, fit_res in results
        # ]
        # # get numerators and denominators out of List[np.ndarray]
        # client_numerators, client_denominators = zip(
        #     *[
        #         from_parameters_ndarrays(params_list_of_ndarrays)
        #         for params_list_of_ndarrays in client_parameters_list_of_ndarrays
        #     ]
        # )

        # # aggregate by summing running numerator and denominator sums
        # numerator = sum(client_numerators)
        # denominator = sum(client_denominators)

        # convert back to List[np.ndarray] then Parameters dataclass to send to clients
        # parameters = fl.common.ndarrays_to_parameters(
        #     to_parameters_ndarrays(numerator=numerator, denominator=denominator)
        # )
        # return parameters, {}
        return None, {}


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # TODO replace with real implementation
        return []


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the
            previously selected and configured clients. Each pair of
            `(ClientProxy, FitRes` constitutes a successful update from one of the
            previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for client updates.

        Returns
        -------
        aggregation_result : Optional[float]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average.
        """
        # TODO replace with real implementation
        return None, {}


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters.

        This function can be used to perform centralized (i.e., server-side) evaluation
        of model parameters.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters: Parameters
            The current (global) model parameters.

        Returns
        -------
        evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """
        # TODO provide real implementation

        # # Write model to disk. No actual evaluation.
        # # server_round=0 is evaluates an initial centralized model.
        # # We don't initialize one, so we do nothing.
        # if server_round > 0:
        #     numerator, denominator = from_parameters_ndarrays(
        #         fl.common.parameters_to_ndarrays(parameters)
        #     )
        #     model = SirModel(numerator=numerator, denominator=denominator)
        #     checkpoint_name = f"model-{server_round:02}.json"
        #     model.save(self.server_dir / checkpoint_name)
        #     logger.info(f"Model checkpoint {checkpoint_name} saved to disk by server.")
        return None


def test_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    # model = SirModel()
    # disease_outcome_df = pd.read_csv(disease_outcome_data_path)
    # preds_format_df = pd.read_csv(preds_format_path, index_col="pid")
    # return TestClient(
    #     cid=cid,
    #     model=model,
    #     disease_outcome_df=disease_outcome_df,
    #     preds_format_df=preds_format_df,
    #     preds_path=preds_dest_path,
    # )
    return TestClient(
        cid=cid,
        person_data_path=person_data_path,
        household_data_path=household_data_path,
        residence_location_data_path=residence_location_data_path,
        activity_location_data_path=activity_location_data_path,
        activity_location_assignment_data_path=activity_location_assignment_data_path,
        population_network_data_path=population_network_data_path,
        disease_outcome_data_path=disease_outcome_data_path,
        client_dir=client_dir,
        preds_format_path=preds_format_path,
        preds_dest_path=preds_dest_path,
    )


@dataclass
class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""
    cid: str
    person_data_path: Path
    household_data_path: Path
    residence_location_data_path: Path
    activity_location_data_path: Path
    activity_location_assignment_data_path: Path
    population_network_data_path: Path
    disease_outcome_data_path: Path
    client_dir: Path
    preds_format_path: Path
    preds_dest_path: Path

    def __post_init__(self):
        """ Start Go client process for processing later """
        self.client = go_start(self.client_dir / GO_EXEC, 'federated-test-client',
            '-person-data-path', path_str(self.person_data_path),
            '-household-data-path', path_str(self.household_data_path),
            '-residence-location-data-path', path_str(self.residence_location_data_path),
            '-activity-location-data-path', path_str(self.activity_location_data_path),
            '-activity-location-assignment-data-path', path_str(self.activity_location_assignment_data_path),
            '-population-network-data-path', path_str(self.population_network_data_path),
            '-disease-outcome-data-path', path_str(self.disease_outcome_data_path),
            '-client-dir', path_str(self.client_dir),
            '-preds-format-path', path_str(self.preds_format_path),
            '-preds-dest-path', path_str(self.preds_dest_path),
        )

    # def __init__(
    #     self,
    #     cid: str,
    #     model: SirModel,
    #     disease_outcome_df: pd.DataFrame,
    #     preds_format_df: pd.DataFrame,
    #     preds_path: Path,
    # ):
    #     super().__init__()
    #     self.cid = cid
    #     self.model = model
    #     self.disease_outcome_df = disease_outcome_df
    #     self.preds_format_df = preds_format_df
    #     self.preds_path = preds_path


    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        # TODO provide real implementation

        # Make predictions on the test split. Use model parameters from server.
        # set_model_parameters(self.model, parameters)
        # predictions = self.model.predict(self.disease_outcome_df)
        # predictions.loc[self.preds_format_df.index].to_csv(self.preds_path)
        # logger.info(f"Client test predictions saved to disk for client {self.cid}.")
        # # Return empty metrics. We're not actually evaluating anything
        return 0.0, 0, {}


def test_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    test_strategy = TestStrategy(server_dir=server_dir)
    num_rounds = 1
    return test_strategy, num_rounds


@dataclass
class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""
    server_dir: Path

    def __post_init__(self):
        """ Start Go strategy process for processing later """
        self.strategy = go_start(self.server_dir / GO_EXEC, 'federated-test-strategy',
            '-server-dir', path_str(self.server_dir),
        )


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Load saved model parameters from training.

        Parameters
        ----------
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        # TODO provide real implementation

        # logger.info("Loading saved model from checkpoint...")
        # last_checkpoint_path = sorted(self.server_dir.glob("model-*.json"))[-1]
        # model = SirModel.load(last_checkpoint_path)
        # parameters_ndarrays = to_parameters_ndarrays(model.numerator, model.denominator)
        # parameters = fl.common.ndarrays_to_parameters(parameters_ndarrays)
        # logger.info(
        #     f"Model parameters loaded from checkpoint {last_checkpoint_path.name}"
        # )
        # return parameters


    def configure_fit(self, server_round, parameters, client_manager):
        """Do nothing and return empty list. We don't need to fit clients for test."""
        # TODO provider real implementation ?
        return []


    def aggregate_fit(self, server_round, results, failures):
        """Do nothing and return empty results. No fit results to aggregate for test."""
        # TODO provider real implementation ?
        return None, {}


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Run evaluate on all clients to make test predictions."""
        # TODO provider real implementation ?
        evaluate_ins = fl.common.EvaluateIns(parameters, {})
        clients = list(client_manager.all().values())
        return [(client, evaluate_ins) for client in clients]


    def aggregate_evaluate(self, server_round, results, failures):
        """Do nothing and return empty results. Not actually evaluating any metrics."""
        # TODO provider real implementation ?
        return None, {}


    def evaluate(self, server_round, parameters):
        # TODO provider real implementation ?
        return None
