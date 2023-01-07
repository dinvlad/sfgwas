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
