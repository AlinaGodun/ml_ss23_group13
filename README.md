# ml_ss23_group13

Create Conda Environment
```
conda env create -f conda_env.yml
```

Activate Conda Environment
```
conda activate ML
```

## Datasets used
- [Autism Screening Adult Data set](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)

- Transformed version of the [Polish companies bankruptcy data set](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#)

## Scripts
- `utils/pb_dataset_creation.py`:
  - Transforms the five .arff files from the original [Polish companies bankruptcy data set](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#) into a single .csv file.
  - Creates a .pickle file that maps the feature names **AttrXX** from the dataset to more descriptive names.
  - Only meant as reference -> will not run as is, as the transformed dataset is included in the repository anyways.