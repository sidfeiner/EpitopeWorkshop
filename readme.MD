# Setup
Assuming you have Anaconda / Miniconda installed, run the following command from the project's 
parent directory: `conda env update -f environment.yml`.

### Pycharm
If you run everything through PyCharm, open the project settings and make sure
it is configured to run with the conda env called `epitope-workshop`. <br />
If you need to browse to the env interpreter, go to  `${CONDA_PREFIX}/envs/epitope-workshop/bin/python`

### CLI / Jupyter
Run `conda activate epitope-workshop` and then `jupyter lab`