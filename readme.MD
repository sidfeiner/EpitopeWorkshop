# Setup
Assuming you have Anaconda / Miniconda installed, run the following command from the project's 
parent directory: `conda env update -f environment37.yml`.

### Pycharm
If you run everything through PyCharm, open the project settings and make sure
it is configured to run with the conda env called `epitope-workshop`. <br />
If you need to browse to the env interpreter, go to  `${CONDA_PREFIX}/envs/epitope-workshop/bin/python`

### CLI / Jupyter
Run `conda activate epitope-workshop` and then `jupyter lab`

# Run
We use Google's `fire` module to turn our code into a CLI. So you must run `main.py`
with all the arguments that the `main` function expects:
* `sequences-file-path` - Path to input file with amino acid sequences. File must be in FASTA format.
* `limit-sequences-amt` - Optional argument to limit amount of records from FASTA file
in case you want a quicker setup during dev.
  
For example: `main.py --sequences_file_path /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/data/iedb_linear_epitopes.fasta --limit-sequences-amt 150`

# Maintain
To facilitate working with the DF's fields, manage all field names in the `common.contract` file
and use those variables in the code anywhere you interact with the DF. 