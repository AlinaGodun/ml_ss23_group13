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

- Transformed version of the [Polish companies bankruptcy data set](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#). We are using the `3year.arff ` file.

## Jupyter notebooks
- `fertility_dataset_exploration.ipynb`:
  - Analysis of fertility dataset. Creates barplots that show the distribution of categorical variables. Creates histogram to show the distribution of numeric variables.
- `bankruptcy_dataset_exploration.ipynb`:
  - Analyis of bankruptcy dataset. Creates histograms, kdeplots and scatterplots. Shows missing value distribution. Performs some basic outlier removal to make most plots easier to interpret and understand.

## Scripts
- `utils/dataset_transformation.py`:
  - Transforms the .arff file from the original [Polish companies bankruptcy data set](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#) into a single .csv file.
  - Creates a .pickle file that maps the feature names **AttrXX** from the dataset to more descriptive names.
  - Only meant as reference -> will not run as is, as the transformed dataset is included in the repository anyways.