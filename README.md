# doj-foia-model-portfolio
This repository serves as a staging environment for the portfolio of all the machine learning models Polydelta experimented with and evaluated with regards to the Department of Justice's FOIA Project.

### Project Description
The Department of Justice, through the Freedom of Information Act Portal on FOIA.gov, supports citizens seeking information from our government. With 118 agencies, citizens must navigate a large, dense set of information to understand if the information they are requesting is already publicly available or to which agency they need to make their request. In order to improve the user experience for navigating this complex process of making FOIA inquiries, reduce inaccurate or incomplete submissions, and direct requests to the correct agencies, Polydelta experimented with and evaluated over 20 different machine learning models. Based on industry standard and custom-made model evaluation metrics, Polydelta recommends developing 2 machine learning models:

1. A Supervised Semantic Similarity model to search for requested information in publicly available documents
2. A Convolutional Neural Network Classification model to recommend agencies to which a request should be submitted


### Installation
In order to properly install the requirements for this project, please follow the instructions below:

Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

In the Terminal, clone the repository:
```
git clone https://github.com/Polydelta-ai/doj-foia-model-portfolio.git
```

In the Terminal, setup a [virtual conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) via the `environment.yml`:
```
conda env create -f environment.yml
```

Verify the environment was installed correctly via:
```
conda env list
```

Activate the environment:
```
conda activate doj-foia-env
```

### Instructions
Test text.
