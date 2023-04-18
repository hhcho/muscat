from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler

from .muscat_model import MusCATModel

NUM_DAYS_FOR_PRED = 2
IMPUTE = True
NEG_TO_POS_RATIO = 3
NUM_EPOCHS = 15
BATCH_SIZE = 1000
USE_ADAM = True
ADAM_LEARN_RATE = 0.005
SGD_LEARN_RATE = 0.01

def load_disease_outcome(cache_dir: Path, disease_outcome_data_path: Path, max_duration: int = -1):
    """
    Utility function that attempts to load cached training
    labels and ID map from disk, or otherwise constructs them.
    """
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

        if max_duration > 0:
            logger.info(f"Clipping max infection duration at {max_duration} (for differential privacy)...")

            durations = (Ytrain == 1).sum(axis=1)
            for idx in np.where(durations > max_duration)[0]:
                start = np.argmax(Ytrain[idx] > 0)
                to_clip = durations[idx] - max_duration
                Ytrain[idx,start:start+to_clip] = 0

        logger.info(f"Saving training labels ({Ytrain.shape}) and id map to cache...")

        np.save(cache_dir / "unique_pids.npy", uid)
        np.save(cache_dir / "disease_outcome_matrix.npy", Ytrain)

    return Ytrain, id_map


def fit(
    person_data_path: Path,
    household_data_path: Path, # not used
    residence_location_data_path: Path, # not used
    activity_location_data_path: Path, # not used
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
):
    """
    Function that fits the model on the provided training data and saves
    the model to disk in the provided directory.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """

    logger.info("Loading data...")
    person = pd.read_csv(person_data_path)
    # hhold = pd.read_csv(household_data_path) # Unused
    # resloc = pd.read_csv(residence_location_data_path) # Unused
    # actloc = pd.read_csv(activity_location_data_path) # Unused
    actassign = pd.read_csv(activity_location_assignment_data_path)
    popnet = pd.read_csv(population_network_data_path)
    Ytrain, id_map = load_disease_outcome(model_dir, disease_outcome_data_path)
    logger.info("...done")

    model = MusCATModel()

    model.setup_features(person, actassign, popnet, id_map=id_map)

    model.fit(Ytrain, NUM_DAYS_FOR_PRED, IMPUTE,
        NEG_TO_POS_RATIO, BATCH_SIZE, NUM_EPOCHS, USE_ADAM,
        ADAM_LEARN_RATE if USE_ADAM else SGD_LEARN_RATE, id_map=id_map)

    model.save(model_dir)

def predict(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    """
    Function that loads the model from the provided directory and performs
    inference on the provided test data. Predictions should match the provided
    format and be written to the provided destination path.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """

    logger.info("Loading in model parameters...")
    model = MusCATModel()
    model.load(model_dir)
    logger.info("...done")

    logger.info("Loading data...")
    person = pd.read_csv(person_data_path)
    # hhold = pd.read_csv(household_data_path) # Unused
    # resloc = pd.read_csv(residence_location_data_path) # Unused
    # actloc = pd.read_csv(activity_location_data_path) # Unused
    actassign = pd.read_csv(activity_location_assignment_data_path)
    popnet = pd.read_csv(population_network_data_path)
    Ytrain, id_map = load_disease_outcome(model_dir, disease_outcome_data_path)

    preds_format_df = pd.read_csv(preds_format_path)

    logger.info("Setting up features...")
    model.setup_features(person, actassign, popnet, id_map=id_map)

    logger.info("Computing predictions...")
    pred = model.predict(Ytrain, id_map=id_map)

    scaler = MinMaxScaler()
    scaler.fit(pred)
    pred = scaler.transform(pred)

    # Ignore people who are already infected or have recovered
    pred[Ytrain[:,-1] == 1] = 0
    pred[Ytrain[:,-1] == 2] = 0

    logger.info("Export predictions")

    id_file = model_dir / "unique_pids.npy"
    uid = np.load(id_file)

    pred_df = pd.Series(
        data=pred.ravel(),
        name="score",
        index=pd.Index(data=uid, name="pid"),
    )

    preds_format_df = pd.read_csv(preds_format_path)

    pred_df.loc[preds_format_df.pid].to_csv(preds_dest_path)

    logger.info("All done")