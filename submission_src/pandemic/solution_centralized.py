from pathlib import Path

from loguru import logger
import pandas as pd

import numpy as np
from scipy.sparse import coo_matrix

from muscat_model import MusCATModel
from sklearn.preprocessing import MinMaxScaler

NUM_DAYS_FOR_PRED = 2
IMPUTE = True
NEG_TO_POS_RATIO = 3
NUM_EPOCHS = 15
BATCH_SIZE = 1000
USE_ADAM = True
ADAM_LEARN_RATE = 0.005
SGD_LEARN_RATE = 0.01

def parse_disease_outcome(disease_outcome_data_path: Path):
    distrain = pd.read_csv(disease_outcome_data_path)
    
    logger.info("Constructing training labels...")
    state_map = {"S":0, "I":1, "R":2}
    I, J = distrain["pid"], distrain["day"]
    V = np.array([state_map[s] for s in distrain["state"]], dtype=np.int8)
    Ytrain = np.array(coo_matrix((V, (I, J)), shape=(distrain["pid"].max() + 1, distrain["day"].max() + 1)).todense())
    Ytrain = Ytrain[:,1:]
    logger.info("...done")

    return Ytrain

def fit(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
):

    logger.info("Loading data...")
    person = pd.read_csv(person_data_path)
    # hhold = pd.read_csv(household_data_path) # Unused
    # resloc = pd.read_csv(residence_location_data_path) # Unused
    # actloc = pd.read_csv(activity_location_data_path) # Unused
    actassign = pd.read_csv(activity_location_assignment_data_path)
    popnet = pd.read_csv(population_network_data_path)
    Ytrain = parse_disease_outcome(disease_outcome_data_path)
    logger.info("...done")

    model = MusCATModel()

    model.setup_features(Ytrain, person, actassign, popnet)

    model.fit(Ytrain, NUM_DAYS_FOR_PRED, IMPUTE, 
        NEG_TO_POS_RATIO, BATCH_SIZE, NUM_EPOCHS, USE_ADAM,
        ADAM_LEARN_RATE if USE_ADAM else SGD_LEARN_RATE)
    
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
    Ytrain = parse_disease_outcome(disease_outcome_data_path)

    preds_format_df = pd.read_csv(preds_format_path)

    logger.info("Setting up features...")
    model.setup_features(Ytrain, person, actassign, popnet)

    logger.info("Computing predictions...")
    pred = model.predict(Ytrain)    

    scaler = MinMaxScaler()
    scaler.fit(pred)
    pred = scaler.transform(pred)

    # Ignore people who are already infected or have recovered
    pred[Ytrain[:,-1] == 1] = 0 
    pred[Ytrain[:,-1] == 2] = 0

    pred_df = pd.Series(
        data=pred.ravel(),
        name="score",
        index=pd.Index(data=np.arange(len(person)), name="pid"),
    )

    preds_format_df = pd.read_csv(preds_format_path)

    logger.info("Saving predictions...")
    pred_df.loc[preds_format_df.index].to_csv(preds_dest_path, index_label="pid")
    logger.info("...done with test predictions.")