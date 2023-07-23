import os
import click
import pickle
from bnlp import NER
from bert import trainer


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("model_type")
def main(data_path, model_path, model_type=None, **kwargs):
    """Trains model to model_path using the data in data_path.

    :param data_path: Path to the directory containing the data files or single file path.
    :param model_path: Path to the directory where the model files should be saved.
    :param model_type: Type of model to train. Can be "bert" or "crf".
    :param kwargs: Additional parameters for the model.
    :return: None.
    """

    if model_type == "bert":
        trainer.train_csebuet_bert_model(data_path, model_path)
    else:
        train_bnlp_crf_model(data_path, model_path)

    print("Done")


def train_bnlp_crf_model(data_path, model_path):

    bn_ner = NER()
    model_name = os.path.join(model_path, "ner_model_trained.pkl")

    with open(os.path.join(data_path, "train", "train.pkl"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, "test", "test.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    bn_ner.train(model_name, train_data, test_data)


if __name__ == '__main__':
    main()
    