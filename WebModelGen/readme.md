# Website Block Labeling

This project presents a website segmentation and segment visual labeling tool, a feature extraction tool and a classifier. Initially the tool is created to classify website saliency. The code can be re-adapted to your experiments by editing small parts of the codes and inputing your own features, and segmentation tools.

## Setup

### Windows:

To setup the project 
just run:

```
pip install -r <project-root>/WebModelGen/requirements.txt
```

### Linux:
To setup the project 
just run:

```
bash setup.sh
```

## Labeling

The labeling tool allows you to visualize blocks of elements on website screenshot and labeling them according to your specific experiments.

To run the tool, run the following command:

```
python <project-root>/label.py
```

> The available arguments:
>
> - `--i` : path to website list. the website list must contain website urls each on one line. The default list is provided under `<project-root>/websites.txt`
> - `--o` : path to output directory.
> - `--segment`: The segmentation generator. For the time being we only support a customized version of Vips. To create or adapt your own, make sure it conforms to the `SegmentationGenerator` class in `<project-root>/crawl/types.py`
> - `--filter`: Manual pre-scoring heuristic method. used to filter trivial blocks out of the classification set. For the time being we only support `saliency_score`.
> - `--min`: The minimum score for a block to stay in the classification set.

The classification tool will generate a block list JSON representation.

## Feature extraction

The feature extraction tool allows you to load the JSON block list and extract features to use in classification. We currently provide 3 categories of features (positional/textual/visual). To add your own features edit `features.py`.

To run the tool, run the following command:

```
python <project-root>/WebModelGen/features.py
```

> The available arguments:
>
> - `--i` : path to a file containing the JSON website dataset.
> - `--o` : path to output directory.

The classification tool will generate a feature CSV file.

## Classifying

The classification tool allows you to do 2 tasks:

- train a classification model on the feature list, and returns a report and the resulting model.
- visualize the output of the model on a chosen page.

To run the training, run the following command:

```
python <project-root>/WebModelGen/classify.py
```

> The available arguments:
>
> - `--data` : path to features file.
> - `--out` : path to output directory.
> - `--save`: save the trained model.
> - `--probability`: Log prediction probabilities.
> - `--interpret`: Log results of tree interpreter.

The classification tool will generate a model `.joblib` file.

To run the testing, run the following command:

```
python <project-root>/WebModelGen/classify.py --debug <url to website> --trained <path to model.joblib>
```

The testing tool will display the positively labeled blocks.
