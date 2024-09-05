import optuna
import subprocess
import yaml

# Load params.yaml to get the default values for the hyperparameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

def objective(trial):
    # Define the search space for hyperparameters
    params['preprocess']['blur_kernel'] = trial.suggest_int('preprocess.blur_kernel', 3, 15)
    params['extract']['hog_orientations'] = trial.suggest_int('extract.hog_orientations', 6, 12)
    params['train']['random_state'] = trial.suggest_int('train.random_state', 1, 100)

    # Save the updated parameters to params.yaml
    with open('params.yaml', 'w') as f:
        yaml.safe_dump(params, f)

    # Run the pipeline
    subprocess.run(['dvc', 'repro', 'train'])

    # Load the evaluation metric from the output file
    with open('outputs/metrics.csv', 'r') as f:
        metric = float(f.read().strip())  # Assuming the metric is a single float value

    return metric  # Minimize or maximize based on your requirement

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Save best parameters
best_params = study.best_params
with open('outputs/tuning_results.csv', 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key},{value}\n")
