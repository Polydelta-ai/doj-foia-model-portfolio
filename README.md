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

Verify the cloning was successful, by running:
'''
cd doj-foia-model-portfolio
'''

In the Terminal, setup a [virtual conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) via the `environment.yml`:
```
conda env create -f environment.yml
```

Verify the environment was installed correctly via:
```
conda env list
```

Install the Jupyter Notebook Kernel:
```
conda install nb_conda_kernels
```

### Instructions
To begin testing the Frequently Requested Document Search and the Agency Recommendation Service models, open the Jupyter Notebook interface via the Terminal, by running:
'''
jupyter notebook
'''

This will open a webpage where you will see the model portfolio's contents. Select either the 'frequently_requested_docs_models.ipynb' or 'agency_finder_models.ipynb' notebooks.

Once the notebook is open, navigate to the 'Kernel' tab at the top of the page, and hover your mouse over 'Change Kernel' and select 'Python[conda env: doj-foia-env].

This may cause a pop up to appear asking you to 'Try and restart the kernel' , select the 'Don't restart' option. this is a known error with Jupyter Notebooks and should not occur after your first time running the application.
