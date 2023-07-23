import click

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(model_path, data_path, output_path):
    """Takes the latest modified model file in the model_path directory,
    Runs evaluation on the data_path directory and saves the results
    to the output_path directory.
    
    :param model_path: Path to the directory containing the model files.
    :param data_path: Path to the directory containing the data files.
    :param output_path: Path to the directory where the results should be saved.

    :Example:
        python test_model.py /path/to/model /path/to/data /path/to/output
    """
    test_bert_model(model_path)


def test_bert_model(model_path):
    import torch
    from transformers import AutoTokenizer
    import numpy as np
    import pickle
    import os

    model_checkpoint = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = torch.load(os.path.join(model_path, "bn_ner_csebuet_bert.pt"))
    with open("models/label_mapping.pkl", 'rb') as f:
        label_mapping = pickle.load(f)
    

    test_sentence = """
    আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম 
    """
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []

    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(label_idx)
            new_tokens.append(token)
            
    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label_mapping[label], token))
    print("Done")



if __name__ == '__main__':
    main()