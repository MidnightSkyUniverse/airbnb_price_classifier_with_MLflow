# Build an ML Pipeline for Short-Term Rental Prices in NYC

This is a project that uses mlflow and hydra to build machine learning pipeline
used to predict rental prices using NYC Airbnb data.



## Table of contents

- [Introduction](#build-an-ML-Pipeline-for-Short-Term-Rental-Prices-in-NYC)
- [Notes to the release](#notes-to-the-release)
- [Preliminary steps](#preliminary-steps)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Pre-existing components](#pre-existing-components)
- [Instructions](#instructions)
  * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  * [Data cleaning](#data-cleaning)
  * [Data testing](#data-testing)
  * [Data splitting](#data-splitting)
  * [Train Random Forest](#train-random-forest)
  * [Optimize hyperparameters](#optimize-hyperparameters)
  * [Select the best model](#select-the-best-model)
  * [Test](#test)
  * [Visualize the pipeline](#visualize-the-pipeline)
  * [Release the pipeline](#release-the-pipeline)
  * [Train the model on a new data sample](#train-the-model-on-a-new-data-sample)
- [Cleaning up](#cleaning-up)

## Notes to the release

### Links
- GitHub: [https://github.com/MidnightSkyUniverse/nd0821-c2-build-model-workflow-starter]
- W&B: [https://wandb.ai/midnightskyuniverse/nyc_airbnb?workspace=user-midnightskyuniverse]

### About the model
Random Forest is used to predict the prices. The model was tested with multiple paramenters,
the config.yaml file contains the values with best MAE:
- max_tfidf_features: 9
- n_estimators: 115 
- max_depth: 14 
- max_features: 0.5
 
### Best MAE
- MEA: 31.052
- MEA in test: 31.432


## Preliminary steps

### Create environment
Using conda or miniconda install virtual environment:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 


### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

### Pre-existing components
In order to simulate a real-world situation, we are providing you with some pre-implemented
re-usable components. While you have a copy in your fork, you will be using them from the original
repository by accessing them through their GitHub link, like:

```python
_ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
```
where `config['main']['components_repository']` is set to 
[https://github.com/udacity/nd0821-c2-build-model-workflow-starter#components](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/tree/master/components).
You can see the parameters that they require by looking into their `MLproject` file:

- `get_data`: downloads the data. [MLproject](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/get_data/MLproject)
- `train_val_test_split`: segrgate the data (splits the data) [MLproject](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/train_val_test_split/MLproject)

## In case of errors
When you make an error writing your `conda.yml` file, you might end up with an environment for the pipeline or one
of the components that is corrupted. Most of the time `mlflow` realizes that and creates a new one every time you try
to fix the problem. However, sometimes this does not happen, especially if the problem was in the `pip` dependencies.
In that case, you might want to clean up all conda environments created by `mlflow` and try again. In order to do so,
you can get a list of the environments you are about to remove by executing:

```
> conda info --envs | grep mlflow | cut -f1 -d" "
```

If you are ok with that list, execute this command to clean them up:

**_NOTE_**: this will remove *ALL* the environments with a name starting with `mlflow`. Use at your own risk

```
> for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

This will iterate over all the environments created by `mlflow` and remove them.


## Instructions

The pipeline is defined in the ``main.py`` file in the root of the starter kit. The file already
contains some boilerplate code as well as the download step. 


### Exploratory Data Analysis (EDA)

1. The ``main.py`` script already comes with the download step implemented. Run the pipeline to 
   get a sample of the data. The pipeline will also upload it to Weights & Biases:
   
  ```bash
  > mlflow run . -P steps=download
  ```
  
  You will see a message similar to:

  ```
  2021-03-12 15:44:39,840 Uploading sample.csv to Weights & Biases
  ```
  This tells you that the data is going to be stored in W&B as the artifact named ``sample.csv``.

2. Now execute the `eda` step:
   ```bash
   > mlflow run src/eda
   ```

3. In jupyter under src/eda there is a file called `EDA.ipynb`
   The file contains EDA basic analyses that are baseline for data cleaning actions
   included in 'basic_cleaning' step. 

    Note that we use ``save_code=True`` in the call to ``wandb.init`` so 
	the notebook is uploaded and versioned by W&B.

4. Terminate the run by running `run.finish()`
5. Save the notebook, then close it (File -> Close and Halt). In the main Jupyter notebook page, 
   click Quit in the upper right to stop Jupyter. This will also terminate the mlflow run. 
   DO NOT USE CRTL-C

## Data cleaning

Now we transfer the data processing we have done as part of the EDA to a new ``basic_cleaning`` 
step that starts from the ``sample.csv`` artifact and create a new artifact ``clean_sample.csv`` 
with the cleaned data:


### Data testing
After the cleaning, it is a good practice to put some tests that verify that the data does not
contain surprises. 

One of our tests will compare the distribution of the current data sample with a reference, 
to ensure that there is no unexpected change. Therefore, we first need to define a 
"reference dataset". We will just tag the latest ``clean_sample.csv`` artifact on W&B as our 
reference dataset. Go with your browser to ``wandb.ai``, navigate to your `nyc_airbnb` project, then to the
artifact tab. Click on "clean_sample", then on the version with the ``latest`` tag. This is the
last one we produced in the previous step. Add a tag ``reference`` to it by clicking the "+"
in the Aliases section on the right:

![reference tag](images/wandb-tag-data-test.png "adding a reference tag")
 
Now we are ready to add some tests. In the starter kit you can find a ``data_tests`` step
that you need to complete. Let's start by appending to 
``src/data_check/test_data.py`` the following test:
  
```python
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000
```
which checks that the size of the dataset is reasonable (not too small, not too large).

Then, add another test ``test_price_range(data, min_price, max_price)`` that checks that 
the price range is between ``min_price`` and ``max_price`` 
(hint: you can use the ``data['price'].between(...)`` method). Also, remember that we are using closures, so the
name of the variables that your test takes in MUST BE exactly `data`, `min_price` and `max_price`.

Now add the `data_check` component to the main file, so that it gets executed as part of our
pipeline. Use ``clean_sample.csv:latest`` as ``csv`` and ``clean_sample.csv:reference`` as 
``ref``. Right now they point to the same file, but later on they will not: we will fetch another sample of data
and therefore the `latest` tag will point to that. 
Also, use the configuration for the other parameters. For example, 
use ``config["data_check"]["kl_threshold"]`` for the ``kl_threshold`` parameter. 

Then run the pipeline and make sure the tests are executed and that they pass. Remember that you can run just this
step with:

```bash
> mlflow run . -P steps="data_check"
```

You can safely ignore the following DeprecationWarning if you see it:

```
DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' 
is deprecated since Python 3.3, and in 3.10 it will stop working
```

### Data splitting
Use the provided component called ``train_val_test_split`` to extract and segregate the test set. 
Add it to the pipeline then run the pipeline. As usual, use the configuration for the parameters like `test_size`,
`random_seed` and `stratify_by`. Look at the `modeling` section in the config file.

**_HINT_**: The path to the step can
be expressed as ``mlflow.run(f"{config['main']['components_repository']}/train_val_test_split", ...)``.

You can see the parameters accepted by this step [here](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/train_val_test_split/MLproject)

After you execute, you will see something like:

```
2021-03-15 01:36:44,818 Uploading trainval_data.csv dataset
2021-03-15 01:36:47,958 Uploading test_data.csv dataset
```
in the log. This tells you that the script is uploading 2 new datasets: ``trainval_data.csv`` and ``test_data.csv``.

### Train Random Forest
Complete the script ``src/train_random_forest/run.py``. All the places where you need to insert code are marked by
a `# YOUR CODE HERE` comment and are delimited by two signs like `######################################`. You can
find further instructions in the file.

Once you are done, add the step to ``main.py``. Use the name ``random_forest_export`` as ``output_artifact``.

**_NOTE_**: the main.py file already provides a variable ``rf_config`` to be passed as the
            ``rf_config`` parameter.

### Optimize hyperparameters
Re-run the entire pipeline varying the hyperparameters of the Random Forest model. This can be
accomplished easily by exploiting the Hydra configuration system. Use the multi-run feature (adding the `-m` option 
at the end of the `hydra_options` specification), and try setting the parameter `modeling.max_tfidf_features` to 10, 15
and 30, and the `modeling.random_forest.max_features` to 0.1, 0.33, 0.5, 0.75, 1.

HINT: if you don't remember the hydra syntax, you can take inspiration from this is example, where we vary 
two other parameters (this is NOT the solution to this step):
```bash
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```
you can change this command line to accomplish your task.

While running this simple experimentation is enough to complete this project, you can also explore more and see if 
you can improve the performance. You can also look at the Hydra documentation for even more ways to do hyperparameters 
optimization. Hydra is very powerful, and allows even to use things like Bayesian optimization without any change
to the pipeline itself.

### Select the best model
Go to W&B and select the best performing model. We are going to consider the Mean Absolute Error as our target metric,
so we are going to choose the model with the lowest MAE.

![wandb](images/wandb_select_best.gif "wandb")

**_HINT_**: you should switch to the Table view (second icon on the left), then click on the upper
            right on "columns", remove all selected columns by clicking on "Hide all", then click
            on the left list on "ID", "Job Type", "max_depth", "n_estimators", "mae" and "r2".
            Click on "Close". Now in the table view you can click on the "mae" column
            on the three little dots, then select "Sort asc". This will sort the runs by ascending
            Mean Absolute Error (best result at the top).

When you have found the best job, click on its name. If you are interested you can explore some of the things we
tracked, for example the feature importance plot. You should see that the `name` feature has quite a bit of importance
(depending on your exact choice of parameters it might be the most important feature or close to that). The `name`
column contains the title of the post on the rental website. Our pipeline performs a very primitive NLP analysis 
based on [TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/) (term frequency-inverse document frequency) and can 
extract a good amount of information from the feature.

Go to the artifact section of the selected job, and select the 
`model_export` output artifact.  Add a ``prod`` tag to it to mark it as 
"production ready".

### Test
Use the provided step ``test_regression_model`` to test your production model against the
test set. Implement the call to this component in the `main.py` file. As usual you can see the parameters in the
corresponding [MLproject](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/test_regression_model/MLproject) 
file. Use the artifact `random_forest_export:prod` for the parameter `mlflow_model` and the test artifact
`test_data.csv:latest` as `test_artifact`.

**NOTE**: This step is NOT run by default when you run the pipeline. In fact, it needs the manual step
of promoting a model to ``prod`` before it can complete successfully. Therefore, you have to
activate it explicitly on the command line:

```bash
> mlflow run . -P steps=test_regression_model
```

### Visualize the pipeline
You can now go to W&B, go the Artifacts section, select the model export artifact then click on the
``Graph view`` tab. You will see a representation of your pipeline.

### Release the pipeline
First copy the best hyper parameters you found in your ``configuration.yml`` so they become the
default values. Then, go to your repository on GitHub and make a release. 
If you need a refresher, here are some [instructions](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release)
on how to release on GitHub.

Call the release ``1.0.0``:

![tag the release](images/tag-release-github.png "tag the release")

If you find problems in the release, fix them and then make a new release like ``1.0.1``, ``1.0.2``
and so on.

### Train the model on a new data sample

Let's now test that we can run the release using ``mlflow`` without any other pre-requisite. We will
train the model on a new sample of data that our company received (``sample2.csv``):

(be ready for a surprise, keep reading even if the command fails)
```bash
> mlflow run https://github.com/[your github username]/nd0821-c2-build-model-workflow-starter.git \
             -v [the version you want to use, like 1.0.0] \
             -P hydra_options="etl.sample='sample2.csv'"
```

**_NOTE_**: the file ``sample2.csv`` contains more data than ``sample1.csv`` so the training will
            be a little slower.

But, wait! It failed! The test ``test_proper_boundaries`` failed, apparently there is one point
which is outside of the boundaries. This is an example of a "successful failure", i.e., a test that
did its job and caught an unexpected event in the pipeline (in this case, in the data).

You can fix this by adding these two lines in the ``basic_cleaning`` step just before saving the output 
to the csv file with `df.to_csv`:

```python
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
```
This will drop rows in the dataset that are not in the proper geolocation. 

Then commit your change, make a new release (for example ``1.0.1``) and retry (of course you need to use 
``-v 1.0.1`` when calling mlflow this time). Now the run should succeed and voit la', 
you have trained your new model on the new data.

## License

[License](LICENSE.txt)
