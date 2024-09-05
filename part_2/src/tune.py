import itertools
from pathlib import Path
from dvc.repo import Repo

# Initialize DVC repository
repo = Repo(".")
data_dir = Path(repo.root_dir, "part_2", "data")
targets = str(Path(repo.root_dir, "part_2", "dvc.yaml"))
print(f"Using DVC repository at '{repo.root_dir}' with data directory at '{data_dir}' and targets file at '{targets}'")

# Define hyperparameter grid
hog_orientations_grid = [9, 12]
lbp_radius_grid = [1, 2]
model_name_grid = ["SVM", "RandomForest"]

# Iterate over all combinations of hyperparameters
for hog_orientations, lbp_radius, model_name in itertools.product(hog_orientations_grid, lbp_radius_grid, model_name_grid):
    # Run experiment with the current set of hyperparameters
    repo.experiments.run(
        queue=True,  # Queue the experiment
        targets=targets,
        #copy_paths=[str(data_dir)],  # Copy the data directory to the experiment
        params=[
            f"extract.hog_orientations={hog_orientations}",
            f"extract.lbp_radius={lbp_radius}",
            f"train.model_name={model_name}",
        ],
    )
    print(f"Experiment queued for HOG orientations={hog_orientations}, LBP radius={lbp_radius}, Model={model_name}")

