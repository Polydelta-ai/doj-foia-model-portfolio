# doj-foia-model-portfolio
This repository serves as a staging environment for the portfolio of all the machine learning models Polydelta experimented with and evaluated with regards to the Department of Justice's FOIA Project.

### Project Description
Polydelta has drawn on its distinctive understanding of public sector technology implementation and program delivery to determine the machine learning models that would best support the federal government-wide effort, led by the Department of Justice (DOJ), to improve the user experience of submitting FOIA requests through the National FOIA Portal (or FOIA.gov). Employing the latest in ML experimentation best practices for this sector, Polydelta evaluated more than XXXX ML models. Based on this evaluation, Polydelta recommends moving forward with two ML models:

    1. Convolutional neural network classification model
    2. Semantic similarity model
    
These two models have been the most accurate in predicting records from publicly available FOIA records/libraries, and identifying the correct agency component when records are not available in the public domain.

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
