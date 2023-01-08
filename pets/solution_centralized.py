'''
Centralized solution based on:
https://github.com/drivendataorg/pets-prize-challenge-runtime/blob/main/submission_templates/pandemic/solution_centralized.py

More information about the data format:
https://www.drivendata.org/competitions/141/uk-federated-learning-2-pandemic-forecasting-federated/page/644
'''
from pathlib import Path

from go import go_run, path_str

GO_EXEC = 'solution'


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
    '''
    go_run(model_dir / GO_EXEC, 'centralized-fit',
        '-person-data-path', path_str(person_data_path),
        '-household-data-path', path_str(household_data_path),
        '-residence-location-data-path', path_str(residence_location_data_path),
        '-activity-location-data-path', path_str(activity_location_data_path),
        '-activity-location-assignment-data-path', path_str(activity_location_assignment_data_path),
        '-population-network-data-path', path_str(population_network_data_path),
        '-disease-outcome-data-path', path_str(disease_outcome_data_path),
        '-model-dir', path_str(model_dir),
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
    go_run(model_dir / GO_EXEC, 'centralized-predict',
        '-person-data-path', path_str(person_data_path),
        '-household-data-path', path_str(household_data_path),
        '-residence-location-data-path', path_str(residence_location_data_path),
        '-activity-location-data-path', path_str(activity_location_data_path),
        '-activity-location-assignment-data-path', path_str(activity_location_assignment_data_path),
        '-population-network-data-path', path_str(population_network_data_path),
        '-disease-outcome-data-path', path_str(disease_outcome_data_path),
        '-model-dir', path_str(model_dir),
        '-preds-format-path', path_str(preds_format_path),
        '-preds-dest-path', path_str(preds_dest_path),
    )
