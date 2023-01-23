import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
                         Parameters, Scalar, Status)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import itertools

from .muscat_model import MusCATModel
from .muscat_workflow import prot, workflow

NUM_DAYS_FOR_PRED = 2
IMPUTE = True
NEG_TO_POS_RATIO = 3
NUM_ITERS = 10
NUM_BATCHES = 10
USE_ADAM = True
ADAM_LEARN_RATE = 0.005
SGD_LEARN_RATE = 0.01

# TRAIN_ROUNDS = workflow.PLAIN_TRAIN_ROUNDS(NUM_ITERS)
# TEST_ROUNDS = workflow.PLAIN_TEST_ROUNDS

# TRAIN_ROUNDS = workflow.SECURE_TRAIN_ROUNDS(NUM_ITERS)
# TEST_ROUNDS = workflow.SECURE_TEST_ROUNDS

TRAIN_ROUNDS = workflow.DEBUG_ROUNDS
TEST_ROUNDS = workflow.EMPTY

def run(*args: str):
    """ Runs Go subprocess with specified args """
    exec_path = os.path.join(os.path.dirname(__file__), 'petchal')
    subprocess.run([exec_path, *args], check=True)

# With original ciphertext data to decrypt in inFile and intermediate
# aggregated data sent from server in enc, finish collective decryption
# keyFile includes cached state from the previous step of this protocol
def collective_decrypt_final_step(client_dir, enc):

    # Files from the previous round (prot.COLLECTIVE_DECRYPT)
    inFile = client_dir / "input.bin" 
    keyFile = client_dir / "decryption_token.bin"

    outFile = client_dir / "output.bin"
    serverFile = client_dir / "server.bin"

    enc.tofile(serverFile)

    run("decrypt-client-receive", client_dir, inFile, serverFile, keyFile, outFile)

    vec = np.fromfile(outFile, dtype=np.float64)

    return vec

# Take a list of ndarrays and transform into 
# encrypted binary data
def encrypt_local_ndarrays(client_dir, data_list):
    arr = np.hstack([np.ravel(v) for v in data_list]).astype(np.float64)
    
    infile = client_dir / "input.bin"
    outfile = client_dir / "output.bin"
    arr.tofile(infile)

    run("encrypt-vec", client_dir, infile, outfile)

    enc = np.fromfile(outfile, dtype=np.int8)

    return enc

def ndarrays_to_fit_configuration(round_num: int, arrays: List[np.ndarray], clients: List[ClientProxy]
) -> List[Tuple[ClientProxy, FitIns]]:
    fit_ins = fl.common.FitIns(fl.common.ndarrays_to_parameters(arrays), {"round": round_num})
    return [(client, fit_ins) for client in clients]

def load_disease_outcome(cache_dir: Path, disease_outcome_data_path: Path):
    
    id_file = cache_dir / "unique_pids.npy"
    mat_file = cache_dir / "disease_outcome_matrix.npy"

    if id_file.exists() and mat_file.exists():
        logger.info("Loading cached training labels and id map...")

        Ytrain = np.load(mat_file)

        uid = np.load(id_file)
        id_map = {v: i for i, v in enumerate(uid)}

    else:

        logger.info("Constructing training labels...")

        distrain = pd.read_csv(disease_outcome_data_path)

        uid = np.unique(distrain["pid"])
        id_map = {v: i for i, v in enumerate(uid)}
        
        state_map = {"S":0, "I":1, "R":2}

        distrain_nz = distrain.loc[distrain["state"] != "S"]

        I, J = distrain_nz["pid"], distrain_nz["day"]
        I = np.array([id_map[v] for v in I])
        V = np.array([state_map[s] for s in distrain_nz["state"]], dtype=np.int8)

        Ytrain = np.array(coo_matrix((V, (I, J)), shape=(len(uid), distrain["day"].max() + 1)).todense())
        Ytrain = Ytrain[:,1:]

        logger.info(f"Saving training labels ({Ytrain.shape}) and id map to cache...")

        np.save(cache_dir / "unique_pids.npy", uid)
        np.save(cache_dir / "disease_outcome_matrix.npy", Ytrain)

    return Ytrain, id_map

def train_setup(server_dir: Path, client_dirs_dict: Dict[str, Path]):
    """
    Perform initial setup between parties before federated training. 

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this 
            directory is required for the trained model to be persisted between
            training and test stages.
        client_dirs_dict (Dict[str, Path]): Dictionary of paths to the directories 
            specific to each client that is available over the simulation. Clients 
            can use these directory for saving and reloading client state. This
            dictionary is keyed by the client ID. 
    """

    client_paths = [str(v) for _,v in client_dirs_dict.items()]

    # TODO uncomment 
    # run("setup", str(len(client_dirs_dict)), str(server_dir), *client_paths)
    

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
        # self.num_examples = pd.read_csv(self.disease_outcome_data_path).shape[0]
        self.num_examples = 1 # TODO: replace with real info

    # TrainClient
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

        round_num = int(config['round'])
        round_prot = TRAIN_ROUNDS[round_num]

        logger.info(f">>>>> TrainClient: fit round {round_num} ({round_prot})")

        if round_prot == prot.UNIFY_LOC:

            actassign = pd.read_csv(self.activity_location_assignment_data_path)

            max_res = actassign.loc[actassign["lid"]>1000000000]["lid"].max() - 1000000001
            max_actloc = actassign.loc[actassign["lid"]<1000000000]["lid"].max()
        
            return [np.array([max_res]), np.array([max_actloc])], 0, {}
        
        elif round_prot in [prot.LOCAL_STATS, prot.LOCAL_STATS_SECURE]:

            max_res, max_actloc = parameters

            logger.info("Save max indicies across clients")

            np.save(self.client_dir / "max_indices.npy", np.array([max_res, max_actloc]))

            logger.info("Load data")

            actassign = pd.read_csv(self.activity_location_assignment_data_path)
            Ytrain, id_map = load_disease_outcome(self.client_dir, self.disease_outcome_data_path)

            logger.info("Compute local disease progression stats")

            model = MusCATModel()
            duration_cnt = model.fit_disease_progression(Ytrain)

            new_vec = np.zeros(10) # Max duration of 10 days
            max_ind = min(len(duration_cnt), len(new_vec))
            new_vec[:max_ind] = duration_cnt[:max_ind]
            duration_cnt = new_vec

            logger.info("Compute local symptom development stats")

            symptom_cnt, recov_cnt = model.fit_symptom_development(Ytrain)

            logger.info("Compute local exposure loads for each training day")

            le_res_act = actassign.loc[actassign["lid"] > 1000000000]
            le_other_act = actassign.loc[actassign["lid"] < 1000000000]

            eloads_pop = (Ytrain == 1).sum(axis = 0)

            eloads_res = np.zeros((Ytrain.shape[1], max_res + 1), dtype=np.float32)

            eloads_actloc = np.zeros((Ytrain.shape[1], max_actloc + 1), dtype=np.float32)
            
            for day in tqdm(range(Ytrain.shape[1])):
                infected = Ytrain[:,day] == 1

                pids = np.array([id_map[v] for v in le_res_act["pid"]])
                val = infected[pids] * le_res_act["duration"]/3600
                I = np.zeros(len(le_res_act), dtype=int)
                J = le_res_act["lid"] - 1000000001
                
                eloads_res[day,:] = coo_matrix((val, (I, J)), shape=(1, max_res + 1)).toarray()[0]
            
                pids = np.array([id_map[v] for v in le_other_act["pid"]])
                val = infected[pids] * le_other_act["duration"]/3600
                I = np.zeros(len(le_other_act), dtype=int)
                J = le_other_act["lid"] - 1
                
                eloads_actloc[day,:] = coo_matrix((val, (I, J)), shape=(1, max_actloc + 1)).toarray()[0]

            logger.info("Exposure load computation finished")
            
            data_list = [duration_cnt, symptom_cnt, recov_cnt, eloads_pop, eloads_res, eloads_actloc]

            if round_prot == prot.LOCAL_STATS_SECURE:
                
                logger.info("Encrypt local disease model parameters and exposure loads")

                enc = encrypt_local_ndarrays(self.client_dir, data_list)

                return [enc], 0, {}

            return data_list, 0, {}

        elif round_prot in [prot.FEAT, prot.FEAT_SECURE]:

            logger.info("Load training labels")

            Ytrain, id_map = load_disease_outcome(self.client_dir, self.disease_outcome_data_path)
            max_res, max_actloc = np.load(self.client_dir / "max_indices.npy")

            if round_prot == prot.FEAT:

                duration_cnt, symptom_cnt, recov_cnt, eloads_pop, eloads_res, eloads_actloc = parameters
            
            elif round_prot == prot.FEAT_SECURE:

                logger.info("Finish decryption of aggregate statistics")

                enc = parameters[0]
                vec = collective_decrypt_final_step(self.client_dir, enc)

                # Unpack data
                idx = 0
                def unpack(V, nelem):
                    ret = V[idx:idx+nelem]
                    idx += nelem
                    return ret

                ndays = Ytrain.shape[1]

                duration_cnt, symptom_cnt, recov_cnt = unpack(vec, 3)

                eloads_pop = unpack(vec, ndays)

                eloads_res = unpack(vec, ndays * (max_res + 1))
                eloads_res = eloads_res.reshape((ndays, max_res + 1)).astype(np.float32)

                eloads_actloc = unpack(vec, ndays * (max_actloc + 1))
                eloads_actloc = eloads_actloc.reshape((ndays, max_actloc + 1)).astype(np.float32)

                del vec # Free memory

            agg_data = {"duration_cnt":duration_cnt,
                        "symptom_cnt":symptom_cnt,
                        "recov_cnt":recov_cnt,
                        "eloads_pop":eloads_pop,
                        "eloads_res":eloads_res,
                        "eloads_actloc":eloads_actloc}

            logger.info("Load person, activity, population network tables")

            person = pd.read_csv(self.person_data_path)
            actassign = pd.read_csv(self.activity_location_assignment_data_path)
            popnet = pd.read_csv(self.population_network_data_path)

            model = MusCATModel()

            logger.info("Fit disease model using aggregate statistics")

            model.fit_disease_progression(Ytrain, agg_data)
            model.fit_symptom_development(Ytrain, agg_data)

            logger.info("Prepare local variables for featurization")

            model.setup_features(person, actassign, popnet, id_map)
            
            infected = (Ytrain == 1).transpose()

            logger.info("Generate training features for each day and sample instances")

            train_feat, train_label = model.get_train_feat_all_days(infected, Ytrain, 
                NUM_DAYS_FOR_PRED, IMPUTE, NEG_TO_POS_RATIO, 
                agg_data, id_map)

            logger.info("Shuffle training data")

            rp = np.random.permutation(train_feat.shape[0])
            train_feat = train_feat[rp]
            train_label = train_label[rp]

            logger.info("Save processed training data")

            np.save(self.client_dir / "train_feat.npy", train_feat)
            np.save(self.client_dir / "train_label.npy", train_label)

            model.num_days_for_pred = NUM_DAYS_FOR_PRED
            model.impute = IMPUTE
            model.weights = np.zeros(1 + train_feat.shape[1])
            model.center = np.zeros(train_feat.shape[1])
            model.scale = np.zeros(train_feat.shape[1])

            logger.info("Save initialized model")

            model.save(self.client_dir, is_fed=True)

            logger.info("Compute feature statistics")

            feat_sum = train_feat.sum(axis=0)
            feat_sqsum = (train_feat**2).sum(axis=0)
            feat_count = train_feat.shape[0]

            data_list = [feat_sum, feat_sqsum, feat_count]
            
            if round_prot == prot.FEAT_SECURE:

                logger.info("Encrypt local feature statistics")

                enc = encrypt_local_ndarrays(self.client_dir, data_list)
                return [enc], 0, {}

            return data_list, 0, {}

        elif round_prot in [prot.ITER, prot.ITER_FIRST, prot.ITER_LAST,
                            prot.ITER_FIRST_SECURE, prot.ITER_SECURE,
                            prot.ITER_LAST_SECURE]:

            
            logger.info("Load model")

            model = MusCATModel()
            model.load(self.client_dir, is_fed=True)
            
            if round_prot in [prot.ITER_FIRST, prot.ITER_FIRST_SECURE]:

                logger.info("First iter: compute feature means and standard deviations")

                if round_prot == prot.ITER_FIRST:

                    feat_sum, feat_sqsum, feat_count = parameters

                elif round_prot == prot.ITER_FIRST_SECURE:

                    logger.info("Finish decryption of feature statistics")

                    enc = parameters[0]
                    vec = collective_decrypt_final_step(self.client_dir, enc)

                    feat_sum, feat_sqsum, feat_count = vec

                model.center = feat_sum / feat_count
                model.scale = np.sqrt((feat_sqsum - (feat_sum**2)) / feat_count)

            else: 

                logger.info("Apply a model update based on previous gradients")
                
                if round_prot in [prot.ITER, prot.ITER_LAST]:
    
                    gradient_sum, sample_count = parameters
                
                elif round_prot in [prot.ITER_SECURE, prot.ITER_LAST_SECURE]:

                    logger.info("Finish decryption of global sum of gradients")

                    enc = parameters[0]
                    vec = collective_decrypt_final_step(self.client_dir, enc)

                    gradient_sum, sample_count = vec[:-1], vec[-1]

                model.weights -= SGD_LEARN_RATE * (gradient_sum / sample_count)

            logger.info("Save model")

            model.save(self.client_dir, is_fed=True)

            # Skip gradient calculation if last iteration
            if round_prot in [prot.ITER_LAST, prot.ITER_LAST_SECURE]:

                logger.info("Training finished")

                return [], 0, {}

            logger.info("Starting gradient computation")

            logger.info("Load data")

            train_feat = np.load(self.client_dir / "train_feat.npy")
            train_label = np.load(self.client_dir / "train_label.npy")
            
            logger.info("Standardize features")

            train_feat -= model.center[np.newaxis,:]
            train_feat /= model.scale[np.newaxis,:]

            logger.info("Sample a batch and compute gradients")

            batch_index = round_num % NUM_BATCHES
            batch_size = int(train_feat.shape[0] / NUM_BATCHES)
            batch_start = batch_index * batch_size
            batch_end = min(train_feat.shape[0], batch_start + batch_size)

            batch_feat = train_feat[batch_start:batch_end]
            batch_label = train_label[batch_start:batch_end]

            gradient_sum, sample_count = model.compute_gradient_sum(batch_feat, batch_label)

            data_list = [gradient_sum, sample_count]

            if round_prot in [prot.ITER_SECURE, prot.ITER_FIRST_SECURE]:

                logger.info("Encrypt local gradients")
            
                enc = encrypt_local_ndarrays(self.client_dir, data_list)

                return [enc], sample_count, {}

            return data_list, sample_count, {}

        elif round_prot == prot.TEST_CPS:
            
            run("cps-test", self.client_dir)

        elif round_prot == prot.TEST_AGG_1:
            
            logger.info(">>>>>>>>>>>" + self.cid)

            if self.cid == "client01":
                intvec = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            else:
                intvec = np.array([5, 2, 0, 3, 10], dtype=np.int64)
            
            enc = encrypt_local_ndarrays(self.client_dir, [intvec])

            return [enc], 0, {}

        elif round_prot in [prot.COLLECTIVE_DECRYPT, prot.TEST_AGG_2]:

            enc = parameters[0]

            infile = self.client_dir / "input.bin"
            outfile = self.client_dir / "output.bin"
            keyfile = self.client_dir / "decryption_token.bin"
            dimfile = self.client_dir / "dimensions.txt"

            logger.info("Save aggregated ciphertext data to disk")

            enc.tofile(infile)

            logger.info("Invoke MHE code for partial decryption")

            # run("decrypt-test", self.client_dir, infile) # For debugging

            run("decrypt-client-send", self.client_dir, infile, outfile, keyfile, dimfile)

            logger.info("Load intermediate decryption results from disk for aggregation")

            enc = np.fromfile(outfile)

            arr = np.loadtxt(dimfile)
            nr, nc = np.round(arr).astype(int)

            return [enc, np.array([nr, nc])], 0, {}

        elif round_prot == prot.TEST_AGG_3:

            enc = parameters[0]

            inFile = self.client_dir / "input.bin"
            outFile = self.client_dir / "output.bin"
            serverFile = self.client_dir / "server.bin"
            keyFile = self.client_dir / "decryption_token.bin"
            enc.tofile(serverFile)

            run("decrypt-client-receive", self.client_dir, inFile, serverFile, keyFile, outFile)

            vec = np.fromfile(outFile, dtype=np.float64)

            logger.info(repr(vec))

        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default return values
        return [], 0, {}

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
    num_rounds = len(TRAIN_ROUNDS) - 1

    logger.info(f">>>>> initialized TrainStrategy, num_rounds: {num_rounds}")
    return training_strategy, num_rounds


@dataclass
class TrainStrategy(fl.server.strategy.Strategy):
    """Federated aggregation equivalent to pooling observations across partitions."""
    server_dir: Path

    def _serialize_file(self, name: str, newbytearray: bytearray):
        with open(name, 'wb') as out:
            out.write(newbytearray)

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

        logger.info(f">>>>> TrainStrategy: initialize_parameters")

        return fl.common.ndarrays_to_parameters([])

    # TrainStrategy
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
        round_num = server_round
        round_prot = TRAIN_ROUNDS[round_num]
        
        logger.info(f">>>>> TrainStrategy: configure_fit round {round_num} ({round_prot})")

        clients = list(client_manager.all().values())
        params = fl.common.parameters_to_ndarrays(parameters)

        if round_prot == prot.UNIFY_LOC:
            pass
        elif round_prot in [prot.LOCAL_STATS, prot.LOCAL_STATS_SECURE]:

            # Provide max_res and max_actloc to all clients
            return ndarrays_to_fit_configuration(round_num, params, clients)

        elif round_prot in [prot.FEAT, prot.FEAT_SECURE]:

            # Provide local disease model stats and exposure loads to all clients
            return ndarrays_to_fit_configuration(round_num, params, clients)

        elif round_prot in [prot.ITER, prot.ITER_FIRST, prot.ITER_LAST]:

            # Provide feature statistics to all clients
            # for main iterations, aggregated gradients
            return ndarrays_to_fit_configuration(round_num, params, clients)

        elif round_prot == prot.COLLECTIVE_DECRYPT:

            # Broadcast aggregated ciphertext data 
            return ndarrays_to_fit_configuration(round_num, params, clients)

        elif round_prot == prot.TEST_AGG_1:
            pass

        elif round_prot == prot.TEST_AGG_2:

            return ndarrays_to_fit_configuration(round_num, params, clients)

        elif round_prot == prot.TEST_AGG_3:

            return ndarrays_to_fit_configuration(round_num, params, clients)

        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default configuration: pass nothing
        return ndarrays_to_fit_configuration(round_num, [], clients)

    # TrainStrategy
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
        metrics : dict
        """
        
        round_num = server_round
        round_prot = TRAIN_ROUNDS[round_num]

        logger.info(f">>>>> TrainStrategy: aggregate_fit round {round_num} ({round_prot})")

        if len(failures) > 0:
            raise Exception(f"Client fit round had {len(failures)} failures.")

        # results is List[Tuple[ClientProxy, FitRes]]
        # convert FitRes to List[List[np.ndarray]]
        fit_res = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if round_prot == prot.UNIFY_LOC:

            max_res = np.array([res[0] for res in fit_res]).max()
            max_actloc = np.array([res[1] for res in fit_res]).max()
            
            params = fl.common.ndarrays_to_parameters([max_res, max_actloc])            
            return params, {}

        elif round_prot == prot.LOCAL_STATS:

            duration_cnt = np.sum([res[0] for res in fit_res], axis=0)
            symptom_cnt = np.sum([res[1] for res in fit_res], axis=0)
            recov_cnt = np.sum([res[2] for res in fit_res], axis=0)
            eloads_pop = np.sum([res[3] for res in fit_res], axis=0)
            eloads_res = np.sum([res[4] for res in fit_res], axis=0)
            eloads_actloc = np.sum([res[5] for res in fit_res], axis=0)

            params = fl.common.ndarrays_to_parameters(
                [duration_cnt, symptom_cnt, recov_cnt, eloads_pop, eloads_res, eloads_actloc])
            return params, {}
            
        elif round_prot == prot.FEAT:
            
            feat_sum = np.sum([res[0] for res in fit_res], axis=0)
            feat_sqsum = np.sum([res[1] for res in fit_res], axis=0)
            feat_count = np.sum([res[2] for res in fit_res], axis=0)

            params = fl.common.ndarrays_to_parameters(
                [feat_sum, feat_sqsum, feat_count])
            return params, {}

        elif round_prot in [prot.ITER, prot.ITER_FIRST]:

            gradient = np.sum([res[0] for res in fit_res], axis=0)
            sample_count = np.sum([res[1] for res in fit_res], axis=0)
            
            params = fl.common.ndarrays_to_parameters(
                [gradient, sample_count])
            return params, {}
        
        elif round_prot == prot.ITER_LAST:
            pass            

        elif round_prot in [prot.FEAT_SECURE, prot.LOCAL_STATS_SECURE, prot.TEST_AGG_1]:

            file_list = []
            for idx, res in enumerate(fit_res):
                enc = res[0]
                fname = self.server_dir / f"input_{idx}.bin"
                enc.tofile(fname)
                file_list.append(fname)
            
            outfile = self.server_dir / "output.bin"

            run("aggregate-cipher", self.server_dir, outfile, *file_list)

            enc = np.fromfile(outfile, dtype=np.int8)

            params = fl.common.ndarrays_to_parameters(
                [enc])
            return params, {}

        elif round_prot in [prot.COLLECTIVE_DECRYPT, prot.TEST_AGG_2]:

            file_list = []
            for idx, res in enumerate(fit_res):
                enc = res[0]
                nr, nc = res[1]
                fname = self.server_dir / f"input_{idx}.bin"
                enc.tofile(fname)
                file_list.append(fname)

            outfile = self.server_dir / "output.bin"

            run("decrypt-server", self.server_dir, str(nr), str(nc), outfile, *file_list)

            enc = np.fromfile(outfile, dtype=np.int8)

            params = fl.common.ndarrays_to_parameters(
                [enc])
            return params, {}

        elif round_prot == prot.TEST_AGG_3:
            pass
        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default return values
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

        logger.info(">>>>> TrainStrategy: configure_evaluate")
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

        logger.info(">>>>> TrainStrategy: aggregate_evaluate")
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

        logger.info(">>>>> TrainStrategy: evaluate")

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

    # TestClient
    def fit(self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:

        round_num = int(config['round'])
        round_prot = TEST_ROUNDS[round_num]

        logger.info(f">>>>> TestClient: fit round {round_num} ({round_prot})")

        if round_prot in [prot.PRED_LOCAL_STATS, prot.PRED_LOCAL_STATS_SECURE]:

            logger.info("Load data")

            max_res, max_actloc = np.load(self.client_dir / "max_indices.npy")

            actassign = pd.read_csv(self.activity_location_assignment_data_path)

            Ytrain, id_map = load_disease_outcome(self.client_dir, self.disease_outcome_data_path)

            logger.info("Compute local exposure loads")

            day = -1 # Last day
            infected = Ytrain[:,day] == 1
            
            eloads_pop = infected.sum()

            pids = np.array([id_map[v] for v in le_res_act["pid"]])
            val = infected[pids] * le_res_act["duration"]/3600
            I = np.zeros(len(le_res_act), dtype=int)
            J = le_res_act["lid"] - 1000000001
            
            eloads_res = coo_matrix((val, (I, J)), shape=(1, max_res + 1)).toarray()[0]
        
            pids = np.array([id_map[v] for v in le_other_act["pid"]])
            val = infected[pids] * le_other_act["duration"]/3600
            I = np.zeros(len(le_other_act), dtype=int)
            J = le_other_act["lid"] - 1
            
            eloads_actloc = coo_matrix((val, (I, J)), shape=(1, max_actloc + 1)).toarray()[0]

            data_list = [eloads_pop, eloads_res, eloads_actloc]

            if round_prot == prot.PRED_LOCAL_STATS_SECURE:
                
                logger.info("Encrypt local exposure loads")

                enc = encrypt_local_ndarrays(self.client_dir, data_list)

                return [enc], 0, {}

            return data_list, 0, {}
        
        elif round_prot in [prot.PRED_FEAT, prot.PRED_FEAT_SECURE]:
            
            logger.info("Load disease outcomes")
            
            max_res, max_actloc = np.load(self.client_dir / "max_indices.npy")
            Ytrain, id_map = load_disease_outcome(self.client_dir, self.disease_outcome_data_path)

            if round_prot == prot.PRED_FEAT:

                eloads_pop, eloads_res, eloads_actloc = parameters

            elif round_prot == prot.PRED_FEAT_SECURE:

                logger.info("Finish decryption of global exposure loads")

                enc = parameters[0]
                vec = collective_decrypt_final_step(self.client_dir, enc)

                # Unpack data
                idx = 0
                def unpack(V, nelem):
                    ret = V[idx:idx+nelem]
                    idx += nelem
                    return ret

                eloads_pop = unpack(vec, 1)
                eloads_res = unpack(vec, max_res + 1).astype(np.float32)
                eloads_actloc = unpack(vec, max_actloc + 1).astype(np.float32)


            agg_data = {"eloads_pop":eloads_pop,
                        "eloads_res":eloads_res,
                        "eloads_actloc":eloads_actloc}

            logger.info("Load person, activity, population network tables")

            person = pd.read_csv(self.person_data_path)
            actassign = pd.read_csv(self.activity_location_assignment_data_path)
            popnet = pd.read_csv(self.population_network_data_path)

            logger.info("Load model")

            model = MusCATModel()
            model.load(self.client_dir, is_fed=True)

            logger.info("Setup local variables for featurization")

            model.setup_features(person, actassign, popnet, id_map)

            logger.info("Compute test features")

            infected = (Ytrain[:,-1] == 1).transpose()

            test_feat = model.get_test_feat(infected, Ytrain, self.num_days_for_pred,
                                            self.impute, agg_data, id_map)

            logger.info("Save test features")

            np.save(self.client_dir / "test_feat.npy", test_feat)
        
        elif round_prot == prot.COLLECTIVE_DECRYPT:

            enc = parameters[0]

            infile = self.client_dir / "input.bin"
            outfile = self.client_dir / "output.bin"
            keyfile = self.client_dir / "decryption_token.bin"
            dimfile = self.client_dir / "dimensions.txt"

            logger.info("Save aggregated ciphertext data to disk")

            enc.tofile(infile)

            logger.info("Invoke MHE code for partial decryption")

            run("decrypt-client-send", self.client_dir, infile, outfile, keyfile, dimfile)

            logger.info("Load intermediate decryption results from disk for aggregation")

            enc = np.fromfile(outfile)

            arr = np.loadtxt(dimfile)
            nr, nc = np.round(arr).astype(int)

            return [enc, np.array([nr, nc])], 0, {}

        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default return values
        return [], 0, {}


    def evaluate(self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[float, int, dict]:
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

        # Make predictions on the test split. Use model parameters from server.
        # set_model_parameters(self.model, parameters)
        # predictions = self.model.predict(self.disease_outcome_df)
        
        logger.info(f">>>>> TestClient: evaluate")

        logger.info("Load model")

        model = MusCATModel()
        model.load(self.client_dir, is_fed=True)

        logger.info("Load test features and data")

        Ytrain, id_map = load_disease_outcome(self.client_dir, self.disease_outcome_data_path)
        test_feat = np.load(self.client_dir / "test_feat.npy")

        logger.info("Standardize features")

        test_feat -= model.center[np.newaxis,:]
        test_feat /= model.scale[np.newaxis,:]

        logger.info("Make predictions")

        pred = (self.weights[0] + feat @ self.weights[1:]).ravel()
    
        scaler = MinMaxScaler()
        scaler.fit(pred)
        pred = scaler.transform(pred)

        # Ignore people who are already infected or have recovered
        pred[Ytrain[:,-1] == 1] = 0
        pred[Ytrain[:,-1] == 2] = 0

        logger.info("Export predictions")

        id_file = cache_dir / "unique_pids.npy"
        uid = np.load(id_file)

        pred_df = pd.Series(
            data=pred.ravel(),
            name="score",
            index=pd.Index(data=uid, name="pid"),
        )

        preds_format_df = pd.read_csv(self.preds_format_path)

        predictions.loc[preds_format_df.pid].to_csv(self.preds_path)
        logger.info(f"Client test predictions saved to disk for client {self.cid}.")
        
        return 0, 0, {}


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
    num_rounds = len(TEST_ROUNDS) - 1
    return test_strategy, num_rounds


@dataclass
class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""
    server_dir: Path

    # Not used
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
        
        logger.info(">>>>> TestStrategy: initialize_parameters")
        
        return Parameters([], '')

    # TestStrategy
    def configure_fit( 
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        round_num = server_round
        round_prot = TEST_ROUNDS[round_num]
        
        logger.info(f">>>>> TestStrategy: configure_fit round {round_num} ({round_prot})")

        clients = list(client_manager.all().values())
        params = fl.common.parameters_to_ndarrays(parameters)

        if round_prot in [prot.PRED_LOCAL_STATS, prot.PRED_LOCAL_STATS_SECURE]:
            pass
        elif round_prot in [prot.PRED_FEAT, prot.PRED_FEAT_SECURE, prot.COLLECTIVE_DECRYPT]:

            # Provide aggregate exposure loads to all clients
            return ndarrays_to_fit_configuration(round_num, params, clients)

        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default configuration: pass nothing
        return ndarrays_to_fit_configuration(round_num, [], clients)

    # TestStrategy
    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict]:
        
        round_num = server_round
        round_prot = TEST_ROUNDS[round_num]

        logger.info(f">>>>> TestStrategy: aggregate_fit round {round_num} ({round_prot})")

        if len(failures) > 0:
            raise Exception(f"Client fit round had {len(failures)} failures.")

        # results is List[Tuple[ClientProxy, FitRes]]
        # convert FitRes to List[List[np.ndarray]]
        fit_res = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if round_prot == prot.PRED_LOCAL_STATS:
            
            eloads_pop = np.sum([res[0] for res in fit_res], axis=0)
            eloads_res = np.sum([res[1] for res in fit_res], axis=0)
            eloads_actloc = np.sum([res[2] for res in fit_res], axis=0)

            params = fl.common.ndarrays_to_parameters(
                [eloads_pop, eloads_res, eloads_actloc])
            return params, {}

        elif round_prot in [prot.COLLECTIVE_DECRYPT, prot.PRED_LOCAL_STATS_SECURE]:
            
            file_list = []
            for idx, res in enumerate(fit_res):
                enc = res[0]
                nr, nc = res[1]
                fname = self.server_dir / f"input_{idx}.bin"
                enc.tofile(fname)
                file_list.append(fname)

            outfile = self.server_dir / "output.bin"

            run("decrypt-server", self.server_dir, str(nr), str(nc), outfile, *file_list)

            enc = np.fromfile(outfile, dtype=np.int8)

            params = fl.common.ndarrays_to_parameters(
                [enc])
            return params, {}

        elif round_prot in [prot.PRED_FEAT, prot.PRED_FEAT_SECURE]:
            pass            
        else:
            logger.info(f"Unimplemented round {round_num} ({round_prot})")

        # Default return values
        return None, {}


    # Not used
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Run evaluate on all clients to make test predictions."""
        
        logger.info(">>>>> TestStrategy: configure_evaluate")
        
        evaluate_ins = EvaluateIns(parameters, {})
        clients = list(client_manager.all().values())
        return [(client, evaluate_ins) for client in clients]

    # Not used
    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Do nothing and return empty results. Not actually evaluating any metrics."""
        
        logger.info(">>>>> TestStrategy: aggregate_evaluate")
        
        return None, {}

    # Not used
    def evaluate(
        self, server_round: int, parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        logger.info(">>>>> TestStrategy: evaluate")

        return None
