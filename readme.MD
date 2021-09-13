# Docker execution

Our Docker image is `sfeiner/epitope-workshop`. It's entrypoint is the `main.py` file. For some of the modules, it's
important to set up a mount between your local file system and the Docker container (for example if you would like to
load a cnn of yours, if you want access to the generated heat maps or plots, and if you would like to run the whole
learning flow based on fasta files on your local file system).

If you do, you MUST mount it to the following path in the container: `/workshop/mnt/data`. Once the mount is created you
will see the following sub-directories:

* `cnn-models` - Your generated cnn models will be stored here
* `heat-maps` - Generated heat maps will be stored here
* `plots` - Plots generated during learning will be stored here


Example when no mount is needed:

`docker run sfeiner/epitope-workshop classify --sequence XXXXXX`

Examples when a mount is needed:
* `docker run --volume /path/to/local/dir:/workshop/mnt/data:rw sfeiner/epitope-workshop run-flow --sequences-files-dir /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/data/iedb-3/ --total-workers 15 --split-file-to-parts-amt 150`
* `docker run --volume /path/to/local/dir:/workshop/mnt/data:rw sfeiner/epitope-workshop classify --sequence XXXXXX --heat-map-name result-heat`

To set the `LOG_LEVEL` env variable, add the `-e` flag like such:

```docker run -e LOG_LEVEL=DEBUG sfeiner/epitope-workshop run-flow --sequences-files-dir /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/data/iedb-3/ --total-workers 15 --split-file-to-parts-amt 150```

More explanation on the exposed modules, under `Local Run`

# Local Setup

Assuming you have Anaconda / Miniconda installed, run the following command from the project's parent
directory: `conda env update -f environment.yml`.

### Pycharm

If you run everything through PyCharm, open the project settings and make sure it is configured to run with the conda
env called `epitope-workshop`. <br />
If you need to browse to the env interpreter, go to  `${CONDA_PREFIX}/envs/epitope-workshop/bin/python`

### CLI / Jupyter

Run `conda activate epitope-workshop` and then `jupyter lab`

# Local Run

We use Google's `fire` module to turn our code into a CLI. So you must run `main.py`, and this module is the entrypoint
for everything we can do. So those are the following modules we expose:

* `split` - module to split a fasta file into multiple fasta files
* `features` - calculates the necessary features for a fasta file and persist resulting dataframe to disk
* `over-balancer` - over balance data files such that half the data has positive labels and half of it has negative
  labels
* `under-balancer` - over balance data files such that half the data has positive labels and half of it has negative
  labels
* `shuffle` - take all our gnerated data and shuffle it to train and test files
* `train` - train data based on train and test directories

In addition, there are the `run-flow` and `classify` methods. The first, runs all of the above sequentially:
Splits the data, calculates it's features, over balances, shuffles and then trains the model. The second (`classify`)
gets as input some sequence and returns the sequence where the uppered amino acids are the ones who have a higher
probability of being in the epitope.

If you want to see detailed logs, you can set the env variable `LOG_LEVEL` to:

* `DEBUG`
* `INFO`
* `WARN`
* `ERROR`

For more explanation, run the module's name and then `--help`, for example `python main.py split --help`, or read the
following:

## Commands

### classify

This command has 3 arguments:

* `sequence` - amino acid sequence
* `heat-map-name` - Optional. If given, png file with heat map will be stored with this name
* `print-proba` - If true, will print the probabilities of each amino acid, to be in the epitope. This will reflect the
  cnn's output.
* `print-precision` - Defaults to 3 and will affect the precision of the probabilities of the amino acids ONLY when
  printing.
* `cnn-name` - Optional. If given, this cnn will be used to classify the sequence

Example: `python main.py classify --sequence mntlllPKPKKDETIQCCTKNNcnr`

### list-cnns

This command prints the available cnns that can be referenced in the `classify` command.

Example: `python main.py list-cnns`

### run-flow

This command combines the whole process of extracting features and building a well-learnt CNN model. Required arguments:

* `sequences_files_dir` - Directory with all fasta files that must be included in the process.

This command get MANY optional arguments that affect the flow of the process. For example:

* normalize numerical features
* you can over-balance (this also synthesizes new data), under-balance or apply no balancing
* define window size
* batch size (for teaching the model)
* amount of `epochs`
* `pos_weight` param to be passed to the loss function
* `weight_decay` param to be passed to the optimizer
* `threshold` that termines from which probability, the amino acid is considered to be part of the epitope
* many more... To find out about them run `python main.py run-flow --help`

## Modules

The modules we expose are

* `split`
* `features`
* `over-balancer`
* `under-balancer`
* `shuffle`
* `train`

To find out more about them, run `python main.py <MODULE> --help`. Those are the modules that `run-flow` uses behind the
scenes so in the average use-case, you will only use `run-flow` and you won't need to use the individual modules.

# Maintain

To facilitate working with the DF's fields, manage all field names in the `common.contract` file and use those variables
in the code anywhere you interact with the DF. There also is a `common.conf` file where all default configuration are
defined.

