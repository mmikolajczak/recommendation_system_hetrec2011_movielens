# Recommendation system for movie reviews data (hetrec2011-movielens)

This project is a recommendation system using users/movies rating data from hetrec2011-movielens dataset.

The data contains info about rated user/movie pairs as well as movie data like genre, actors that played in it, director, etc.

Recommendation criterion is following: recommend a movie to user
if he/she would rate it above average rating of all movies.

Several models/algorithms are used for recommendation: logistic regression, FFM, wide&deep NN.

## Getting Started

### Prerequisites

##### Hetrec movielens data

Data used for experiments comes from Hetrec 2011 movielens dataset, specifically hetrec2011-movielens-2k-v2. It can be found in a couple of places,
starting with this link https://grouplens.org/datasets/hetrec-2011/

##### LibFFM

This project uses FFM implementation that can be found here: https://github.com/guestwalk/libffm<br/>
To run FFM experiments you will have compile binaries from that repository (authors provides makefiles).
Compiled binaries are later used by simple Python wrapper that can be found in <i>ffm</i> subpackage of project.

### Installing

First thing to do is of course to clone this project to your local repository. You can do that by using following command:

```
git clone https://github.com/mmikolajczak/recommendation_system_hetrec2011_movielens some_fancy_local_path
```

All additional packages are listed in requirements.txt and can be installed by:

```
pip install -r requirements.txt
```

Just be aware in the last position in that file - it is Tensorflow with GPU support installation. If you don't posses GPU that
can be used for computation simply replace the last position with tensorflow. If you do, you will probably need
to install/configure CUDA/CUDANN on your machine - please refer to installation guide on tensorflow.org then.

## Running the experiments

### Data preparation

Before running the experiments the data must be combined from multiple files into some proper input format.
<i>data_preparation_scripts</i> subpackage contains some scripts to do that. For particular
experiments use them as follows:
- Model: <b>FFM</b>,<br/>
Format: <b>.fmm</b>, <br/>
use <i>ffm_data_gen.py</i> (this may take a while (hours even, depending on used featurs and CPU used for generation))
and later cv_split to divide data.

- Model: <b>wide&deep</b> (tensorflow estimator API version),<br/>
Format: <b>.csv</b>,<br/>
use <i>csv_cv_split</i> function from <i>cv_split.py</i>.<br/>
Additionaly, possible data vocabulary is needed for NN, to generate it use <i>generate_categories_vocabularies.py</i> script.


- Model: <b>wide&deep</b> (keras implementation, with some improvements/customization impossible in estimator),<br/>
Format <b>.pkl</b> pickled numpy arrays corresponding to csv data,<br/>
use <i>np_cv_split</i> from <i>cv_split.py</i>
Additionaly, possible data vocabulary is needed for NN, to generate it use <i>generate_categories_vocabularies.py</i> script.

Note: Make sure that data files encoding is coherent with generation scripts, by
default they use <i>utf-8</i>, but downloaded data may use encoding like <i>latin-1</i>.


### Experiments scripts

Experiemnts itselves are located in <i>experiments_scripts</i> subpackage, in files in format <i>{model_name}_experiment.py</i>
All training hyperparameters can be customized in that scripts, but change features/architecture itself for <b>wide&deep</b> one must go
corresponding package. By default, CV is used only in case of <b>FFM</b> due to time constraints.

## Results

There won't be detailed table here but performance between different methods/models varies. Overall, best resutls for all
of them are rather achieved using all possible features. To give overall idea, here are the best results from performed experiments
for each method (in termes of achieved <b>AUC</b> on evaluation set):<br/><br/>

| LogReg | FFM        | wide&deep (tf/Estimator)| wide&deep (keras)|
|:-------------:|:-------------:|:-------------:|:-----:|
| 0,801 | 0,8213    | 0,8074 | 0,8115 |


## Contributing

Potential PRs are most welcomed.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

