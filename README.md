# ML Pipeline for Short-Term Rental Prices in NYC

Here is the Machine Learning project I've worked on for Udacity DevOps training.
We aim to predict rental prices in NYC based on Airbnb data. 

For this project I updated the code provided with the training so all steps of 
the machine learning pipeline could be executed. I worked on EDA analyses,
and with extensive model trianing for two different clean_data subsets,
I decided to performed deeper data cleaning than initially suggested as this 
gave me slightly better model.

Once the model was trained and validated with various parameters, I've selected the best performing model 
and included its parameters in `config.yaml` file.  
The model trained is Random Forest.

Weights & Biases are used as repository of artifacts and runs.


### Links
- GitHub: [https://github.com/MidnightSkyUniverse/nd0821-c2-build-model-workflow-starter]
- W&B: [https://wandb.ai/midnightskyuniverse/nyc_airbnb?workspace=user-midnightskyuniverse]
 
### Best MAE
- MAE: 32.462
- Test MAE: 32.667


## Table of contents

- [Introduction](#build-an-ML-Pipeline-for-Short-Term-Rental-Prices-in-NYC)
- [Preliminary steps](#preliminary-steps)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
- [Instructions](#instructions)
  * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  * [Data cleaning](#data-cleaning)
  * [Data testing](#data-testing)
  * [Data splitting](#data-splitting)
  * [Train Random Forest](#train-random-forest)
  * [Optimize hyperparameters](#optimize-hyperparameters)
  * [Select the best model](#select-the-best-model)
  * [Test](#test)
  * [Train the model on a new data sample](#train-the-model-on-a-new-data-sample)
- [Cleaning up](#cleaning-up)




## Preliminary steps

### Create environment
Using conda or miniconda install virtual environment:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
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
The `main.py` is written so that the steps are defined at the top of the file, in the 
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


## Instructions

The pipeline is defined in the ``main.py`` file in the root of the starter kit. 

### Exploratory Data Analysis (EDA)

1. EDA
EDA analyses are in jupyter notebook. This notebook is our sandbox where we test
our hypotesis and learn about the data. Whatever we decide about the data, is getting implemented
in Data Cleaning step. 
Run the pipeline to get a sample of the data:
   
  ```bash
  > mlflow run . -P steps=download
  ```

2. Now execute the `eda` step:
   ```bash
   > mlflow run src/eda
   ```

## Data cleaning

Now we transfer the data processing we have done as part of the EDA to a new ``basic_cleaning``.


### Data testing

Before you decide to test the data after data cleaning step, make sure you add a tag to the artifact
"clean_sample" with the ``latest`` tag. Add a tag ``reference`` to it by clicking the "+"
in the Aliases section on the right:

![reference tag](images/wandb-tag-data-test.png "adding a reference tag")
 

```bash
> mlflow run . -P steps="data_check"
```

You can safely ignore the following DeprecationWarning if you see it:

```
DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' 
is deprecated since Python 3.3, and in 3.10 it will stop working
```

### Data splitting

This step is uploading 2 new datasets: ``trainval_data.csv`` and ``test_data.csv``.

### Train Random Forest

```bash
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```
you can change this command line to accomplish your task.


### Test
Use the provided step ``test_regression_model`` to test your production model against the
test set. 

This step is NOT run by default when you run the pipeline. In fact, it needs the manual step
of promoting a model to ``prod`` before it can complete successfully. Therefore, you have to
activate it explicitly on the command line:

```bash
> mlflow run . -P steps=test_regression_model
```


### Train the model on a new data sample

Let's now test that we can run the release using ``mlflow`` without any other pre-requisite. We will
train the model on a new sample of data that our company received (``sample2.csv``):

(be ready for a surprise, keep reading even if the command fails)
```bash
> mlflow run https://github.com/midnightskyuniverse/nd0821-c2-build-model-workflow-starter.git \
             -v [the version you want to use, like 1.0.0] \
             -P hydra_options="etl.sample='sample2.csv'"
```


## License

[License](LICENSE.txt)
