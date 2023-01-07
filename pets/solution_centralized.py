'''
Federated solution based on:
https://github.com/drivendataorg/pets-prize-challenge-runtime/blob/main/examples_src/pandemic/solution_centralized.py
'''
from pathlib import Path

from go import go_run

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
    '''
    Fit your model on the training data and write your model to disk.
    '''
    go_run('fit',
        '-person-data-path', person_data_path.as_posix(),
        '-household-data-path', household_data_path.as_posix(),
        '-residence-location-data-path', residence_location_data_path.as_posix(),
        '-activity-location-data-path', activity_location_data_path.as_posix(),
        '-activity-location-assignment-data-path', activity_location_assignment_data_path.as_posix(),
        '-population-network-data-path', population_network_data_path.as_posix(),
        '-disease-outcome-data-path', disease_outcome_data_path.as_posix(),
        '-model-dir', model_dir.as_posix(),
    )


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
    '''
    Load your model and perform inference on the test data.
    '''
    go_run('predict',
        '-person-data-path', person_data_path.as_posix(),
        '-household-data-path', household_data_path.as_posix(),
        '-residence-location-data-path', residence_location_data_path.as_posix(),
        '-activity-location-data-path', activity_location_data_path.as_posix(),
        '-activity-location-assignment-data-path', activity_location_assignment_data_path.as_posix(),
        '-population-network-data-path', population_network_data_path.as_posix(),
        '-disease-outcome-data-path', disease_outcome_data_path.as_posix(),
        '-model-dir', model_dir.as_posix(),
        '-preds-format-path', preds_format_path.as_posix(),
        '-preds-dest-path', preds_dest_path.as_posix(),
    )
