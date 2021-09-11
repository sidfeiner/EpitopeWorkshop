# Local Setup
Assuming you have Anaconda / Miniconda installed, run the following command from the project's 
parent directory: `conda env update -f environment.yml`.

### Pycharm
If you run everything through PyCharm, open the project settings and make sure
it is configured to run with the conda env called `epitope-workshop`. <br />
If you need to browse to the env interpreter, go to  `${CONDA_PREFIX}/envs/epitope-workshop/bin/python`

### CLI / Jupyter
Run `conda activate epitope-workshop` and then `jupyter lab`

# Local Run

We use Google's `fire` module to turn our code into a CLI. So you must run `main.py`, and this module
is the entrypoint for everything we can do. So those are the following modules we expose:
* `split` - module to split a fasta file into multiple fasta files
* `features` - calculates the necessary features for a fasta file and persist resulting dataframe to disk
* `over_balancer` - over balance data files such that half the data has positive labels and half of it has negative labels
* `shuffle` - take all our gnerated data and shuffle it to train and test files
* `train` - train data based on train and test directories
* `sequences-file-path` - Path to input file with amino acid sequences. File must be in FASTA format.
* `limit-sequences-amt` - Optional argument to limit amount of records from FASTA file 
  in case you want a quicker setup during dev.
  
For example: `main.py --sequences_file_path ./data/iedb_linear_epitopes.fasta --limit-sequences-amt 150 --partitions-amt 30`

# Maintain
To facilitate working with the DF's fields, manage all field names in the `common.contract` file
and use those variables in the code anywhere you interact with the DF.