![name-shield]
[![LinkedIn][linkedin-shield]][linkedin-url]


## ML Pipeline for Short-Term Rental Prices in NYC

Here is the Machine Learning project I've worked on for Udacity DevOps training.
We aim to predict rental prices in NYC based on Airbnb data. 

For this project I updated the code provided with the training so all steps of 
the machine learning pipeline could be executed. I worked on EDA analyses,
and with extensive model trianing for two different clean_data subsets,
I decided to performed deeper data cleaning than initially suggested as this 
gave me slightly better model.

Once the model was trained and validated with various parameters, I've selected the best performing model 
and included its parameters in `config.yaml` file. The model is based Random Forest algorithm.

Weights & Biases are used as repository of artifacts and runs.


### Links
- GitHub: [https://github.com/MidnightSkyUniverse/nd0821-c2-build-model-workflow-starter]
- W&B: [https://wandb.ai/midnightskyuniverse/nyc_airbnb?workspace=user-midnightskyuniverse]
 
### Model performance
- mae: 32.462
- test mae: 32.667


## Table of contents

- [Introduction](#build-an-ML-Pipeline-for-Short-Term-Rental-Prices-in-NYC)
- [Preliminary steps](#preliminary-steps)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
- [Instructions](#instructions)
  * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Train Random Forest](#train-random-forest)
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






## Instructions

The pipeline is defined in the ``main.py`` file in the root of the starter kit. 

### Exploratory Data Analysis (EDA)

1. EDA analyses can be found in jupyter notebook. This is where we test our hypotesis and learn about the data. 
Whatever we decide about the data, is getting implemented in Data Cleaning step. 

Run the pipeline to get a sample of the data:
   
  ```bash
  > mlflow run . -P steps=download
  ```

2. Now execute the `eda` step:
  ```bash
  > mlflow run src/eda
  ```

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

Before you decide to test the data after data cleaning step, make sure you add a tag to the artifact
"clean_sample" with the ``latest`` tag. Add a tag ``reference`` to it.

 

```bash
> mlflow run . -P steps="data_check"
```

### Train Random Forest

Before we can train the model, we execute the step to split the data
```bash
> mlflow run . -P steps='data_split'
```

Next we train the model. I used hydra/launcher=joblib option that allows me to run 
as many parallel threats as there is CPUs on host machine

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="hydra/launcher=joblib \
     modeling.max_tfidf_features=9,10 \
     modeling.random_forest.max_features=0.33,0.5 \
     modeling.random_forest.n_estimators=101,108,115 \
     modeling.random_forest.max_depth=12,14 -m"
```



### Test
Use the provided step ``test_regression_model`` to test your production model against the
test set. This step is NOT run by default when you run the pipeline. That test needs the manual step
of promoting a model to ``prod`` before it can complete successfully. 

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

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[name-shield]: https://img.shields.io/badge/Author-Ali%20Binkowska-blueviolet?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alibinkowska
