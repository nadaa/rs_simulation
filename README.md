## Table of content

- [Summary](#summary)
- [General model workflow](#general-model-workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the model](#running-the-model)
- [File structure](#file-structure)
- [Rating dataset](#ratings-dataset)
- [Configuration file](#configuration-file)
- [Results](#results)

## Summary

This agent-based simulation model demonstrates the consequences of various recommendation strategies for different stakeholders: Focusing only on satisfing consumers when delivering the recommendations may affect other stakeholders' interests, in particular the short-term profit of the service provider. Likewise, delivering recommendations only to maximize profit may negatively affect the consumers' trust in the service provider.

Two types of agents are used in the model:

<ul>
<li> Recommendation service provider: Prepares and sends personalized recommendations to the consumers </li>
<li> Consumer: Receive the recommendations and make further decisions </li>
</ul>

## General model workflow based on [Experience goods](https://link.springer.com/chapter/10.1007/978-3-8350-9580-9_1)

![model_workflow](figures/modelgeneralflow.png)

## Requirements

We tested the code on a machine with MS Windows 11, Python=3.10, 16GB, and an Intel Core 7 CPU. The code also was tested using a machine with Docker, Ubuntu 20.04.2 LTS x86_64, , 30GB, and an Intel Xeon E5645 (12) @ 2.4. processor. \
For installation without Docker, it is recommended to install the last version of Anaconda, which comes with Python 3 and supports scientific packages.

The following packages are used in our model, see also the file `requirements.txt`:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)
- [surprise](http://surpriselib.com/)
- [mesa](https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html)
- [pyyaml](https://pyyaml.org/)

## Installation

### Setting up the environment (No Docker)

Download and install [Anaconda](https://www.anaconda.com/products/individual-d) (Individual Edition)

Create a virtual environment

```
conda create -n myenv python=3.8
```

Activate the virtual environment

```
conda activate myenv
```

More commands regarding the use of virtual environments in Anaconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Install the required packages by running:

```
pip install -r requirements.txt
```

If you face errors when insatlling the **surprise** package on MS Windows, run:

```
conda install -c conda-forge scikit-surprise
```

## Running the model

To run the simulation when Docker does not exist:

```
cd src
```

`python run.py`

### Setting up the environment (Using Docker)

We provide a Docker image on Docker hub; to pull the image use the following:

```
docker pull nadadocker/rs_simulation
```

Since the simulation saves data to the disk at the end, an output directory has to be provided to the Docker image. The following command runs a new container of the simulation and saves the output in the "results" directory. Before running the Docker container, create a directory named `results` on the host machine by executing the following commands:

```
mkdir results
```

Run the Docker container.

```
docker run -it --rm -v ${PWD}/results:/results --name <my_container> <nadadocker/rs_simulation>
```

- `container_name`: A name of the container.
- `${PWD}`: The current working directory.
- `-v ${PWD}/results:/results`: Sets up a bind mount volume that links the `/results` directory from inside the 'container_name' to the directory ${PWD}/results on the host machine. Docker uses ':' to split the host’s path from the container path, and the host path always comes first.
- `<nadadocker/simulation>` : The Docker image that is used to run the container.

## File structure

The simulation is built with the help of [Mesa](https://github.com/projectmesa/mesa), an agent-based simulation framework in Python.

```
├── data/
│   ├── dataset                 <- MovieLens dataset
│   │   ├── movies.csv
│   │   └── ratings.csv
│   ├── recdata/                  <- Recommendation algorithm output saved in  pickle format
│   │   ├── consumers_items_utilities_predictions.p
│   │   ├── consumers_items_utilities_predictions_popular.p
│   │   └── SVDmodel.p
│   └── trust/                    <- Initial data for consumer trust
│       └── beta_initials.p
├── Dockerfile
├── figures/                      <- Figures that show simulation results
│   ├── modelgeneralflow.png
│   ├── time-consumption_probability.png
│   ├── time-total_profit.png
│   └── time-trust.png
├── README.md
├── requirements.txt
├── results-analysis(R)/                      <- R code to analyze model output, the output is stored in "results" folder, we store it in a seafile service
├── src/
  ├── __init__.py
  ├── config.yml                  <- Simulation settings
  ├── consumer.py                 <- Contains all propoerties and behaviors of consumer agents
  ├── mesa_utils/
  │   ├── __init__.py
  │   ├── datacollection.py
  │   └── schedule.py
  ├── model.py                    <- Contains the model class, which manages agent creation, data sharing, and simulation output collection
  ├── plots.py                    <- Plotting module for data analysis
  ├── read_config.py
  ├── run.py                      <- Launches the simulation
  ├── service_provider.py         <- Contains all properties and behavior of the service provider agent
  ├── test.py
  └── utils.py             <- An auxiliary module

```

## Ratings dataset

We use the [MovieLens dataset](https://grouplens.org/datasets/movielens/), the small version (1 MB), which contains movie ratings for multiple consumers, [more details](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). The following shows the content of `ratings.csv`.

| userId | movieId | rating | timestamp |
| ------ | ------- | ------ | --------- |
| 1      | 1       | 4      | 964982703 |
| 1      | 3       | 4      | 964981247 |
| 1      | 6       | 4      | 964982224 |
| 1      | 47      | 5      | 964983815 |
| 1      | 50      | 5      | 964982931 |

The dataset is used to predict consumer items utilities, and to initialize the model.

## Configuration file

`config.yml` includes all the required parameters to set up the model.

**Note**: Running the code may take a long time (e.g. one hour) based on the predefined time steps and the number of replications in the configuration.

## Results

Each execution of the model generates a unique folder inside the results folder. The collected data from the simulation contains various CSV files. The data is collected from the simulation in a csv format, and `R language` is used for better visualization.

### Installation

- [R](https://cran.r-project.org/mirrors.html) (required)
- [RStudio](https://posit.co/download/rstudio-desktop/#download) (optional)

### Analyze the data

To create the figures, run the `main.R` script in folder `results-analysis(R)`. You can either run the script in the RStudio or run the script in the command line using R.
When running, a popup window will appear to select the path of the data.
