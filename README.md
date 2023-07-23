Bangla NER
==============================

Bangla Name Entity Recognition Experiments

In this repo I've done some R&D on Bangla Named Entity Recognition (focusing on person name entities).
The annotated dataset used are:
* https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data
* https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl

The first source is pretty clean dataset but the second one need some processing & cleaning to be usable.

With some cleaning and removing data on the second one the total sentences size was around 10k from which 80% data was used to train and 20% for tesing. No validation set was used for time constraints.

The approaches for training the model was,
    
a. CRF based ML approach taken from BanglaNLP. [link](https://bnlp.readthedocs.io/en/latest/)
    
b. Bangla Bert pretrained deeplearning based model finetuing.[link](https://huggingface.co/csebuetnlp/banglabert)


a. The CRF based approach uses some basic features like the suffix & prefix of the word, is first word of the sentence, previous word, next word etc.

There were total of 17 features used but unfortunately 4 of them was about upper case lower case based feature. As Bangla is an uncased language those 4 features does not provide meaningful input for the model.

However the model performs quite well despite such basic features as CRF modes can use meaningful features from the sequence info the the model shows reasonably good performance.

The test set accuracy was:

```
Train set: 8036
Test set: 2010
Accuracy on test data is: 
0.918659853043452
F1 Score(micro) is: 
0.918659853043452
```

But when used only the dataset 1 (first link) which is relatively clean the accuracy was better,
```
Train set: 5260
Test set: 1315
Accuracy on test data is: 
0.944655704008222
F1 Score(micro) is: 
0.944655704008222
```

With more time the issues with the second dataset could be resolved but in this experiment we used the data as is with some basic cleaning.


b. I also added the pretrained BERT base model finetuing. I didn't get a very good accuracy using CSE BUET Bangla BERT pretrained model. The accuracy was around 0.91 which is not satisfactory result for the large model. I couldn't debug the actual cause of the issue but scripts are here in this repo.

Remarks:
------------

I've invested almost 30 hours on this experiment.

I learnt (5-10 hours) about,

* BOI, IOBES etc tagging formats,
* Named entity recognition basics [source1](https://www.kaggle.com/code/eneszvo/ner-named-entity-recognition-tutorial) [source2](https://www.kaggle.com/code/shoumikgoswami/ner-using-random-forest-and-crf)
* Bnlp, BUET Bangla BERT projects etc works on Bangla NLP projects

Worked on (20 hours),

* Code base structure [3 hour]
* Data sync, dataset generation, cleaning, training and testing scripts etc [15 hour]
* Readme reporting [2 hour]


Future works:

* Proper EDA
* CRF model different feature combinations
* Automatic searching for best model using ensemble methods and cross validation.
* Debugging & improving the code quality on BERT based model
* Proper validation using a validation set'
* Experiment using LSTM based shallow models for features
* Resolve class imbalance as there's a large number of 'O' tags
* Focus only on PERSON NER
* etc.


Usage
------------

There is a notebook in the notebooks directory containing the commands.


To sync raw data

```$ make sync_data```

To prepare data from raw data directory

```$ make dataset```

To train model

```$ make train_model```

To test model

```$ make test_model```

To visualize model outputs

```$ make visualize```


Deployment (Not Complete)
------------

To deploy API server

```$ docker compose up -d```

The above command will use the latest (modification time) model from the `models/`



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── configs            <- Configuration files
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries    
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


