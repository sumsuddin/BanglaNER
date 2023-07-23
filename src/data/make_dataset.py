# -*- coding: utf-8 -*-
import os
import click
import logging
import pickle
from glob import glob
try:
    from src.data.utils import delete_files_and_directories
except:
    from utils import delete_files_and_directories

import random


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info("making processed data set from raw data")

    delete_files_and_directories(output_filepath)
    
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    os.makedirs(os.path.join(output_filepath, "train"))
    os.makedirs(os.path.join(output_filepath, "test"))

    dataset = read_dataset(input_filepath)
    print(len(dataset))
    tag_counts = count_tags(dataset)
    dataset = process_data(dataset, tag_counts)

    random.seed(1)
    random.shuffle(dataset)

    train_dataset = dataset[:int(len(dataset)*0.8)]
    test_dataset = dataset[int(len(dataset)*0.8):]

    save_pickle(train_dataset, os.path.join(output_filepath, "train", "train.pkl"))
    save_pickle(test_dataset, os.path.join(output_filepath, "test", "test.pkl"))

    logger.info("Making processed data set from raw data done")


def read_dataset(input_filepath):
    txt_file_dataset = get_dataset_from_txt(input_filepath)
    json_file_dataset = [] # get_dataset_from_json(input_filepath)

    return txt_file_dataset + json_file_dataset


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def count_tags(dataset):
    tags = {}
    for sentence in dataset:
        for word, tag in sentence:
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1
    return tags


def process_data(dataset, tag_counts):
    processed_dataset = []
    for sentence in dataset:
        take = True
        for _, tag in sentence:
            if tag_counts[tag] < 5:
                take = False
        if take:
            processed_dataset.append(sentence)
    return processed_dataset


def get_dataset_from_txt(input_path: str):
    dataset = []
    txt_files = glob(os.path.join(input_path, "*.txt"))
    txt_files = sorted(txt_files)
    for txt_file in txt_files:
        dataset += read_txt_data(txt_file)
    
    
    return dataset

def get_dataset_from_json(input_path: str):
    dataset = []
    json_files = glob(os.path.join(input_path, "*.jsonl"))
    json_files = sorted(json_files)
    for json_file in json_files:
        dataset += read_json_data(json_file)
    return dataset


def read_txt_data(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename,encoding="utf-8-sig")
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" or line[1]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        
        line = line.strip()
        splits = line.split('\t')
        if (splits[-1]=='\n'):
            continue
        if splits[-1] == "TIM":
            splits[-1] = "B-TIM"

        sentence.append([splits[0],splits[-1]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences

def read_jsonl(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            yield json.loads(line)


def split_punctuations(word, punctuations):
    if len(word) == 1:
        return [word]
    
    if "-" in word:
        ind = word.index("-")
        first_part = word[:ind]
        second_part = word[ind+1:]
        tokens = []
        if len(first_part) > 0:
            tokens += split_punctuations(first_part, punctuations)
        tokens.append('-')
        if len(second_part) > 0:
            tokens += split_punctuations(second_part, punctuations)
        return tokens

    if word[0] in punctuations:
        return [word[0]] + split_punctuations(word[1:], punctuations)
    
    if word[-1] in punctuations:
        return split_punctuations(word[:-1], punctuations) + [word[-1]]
    
    return [word]


def word_tokenizer(text: str):
    punctuations = ["।", ",", ";", ":", "?", "!", "'", "”", "\"", "-",
                "[", "]", "{", "}", "(", ")", '–', "—", "―", "~"]
    punctuations = set(punctuations).difference({"-"})

    tokens = [i for i in text.split()]
    final_tokens = []

    for i in tokens:
        word = i.strip()
        tokens = split_punctuations(word, punctuations)
        for token in tokens:
            final_tokens.append(token)

    return final_tokens            

def clean_text(text: str):
    text = text.strip()
    text = text.replace("—", "-")
    text = text.replace("―", "-")
    text = text.replace("–", "-")
    return text

def map_tags(tags):
    mapped_tags = []

    tag_map = {
        "U-PERSON": "B-PER",
        "B-PERSON": "B-PER",
        "I-PERSON": "I-PER",
        "L-PERSON": "I-PER",
        "U-GPE": "B-LOC",
        "B-GPE": "B-LOC",
        "I-GPE": "I-LOC",
        "L-GPE": "I-LOC",
        "U-ORG": "B-ORG",
        "B-ORG": "B-ORG",
        "I-ORG": "I-ORG",
        "L-ORG": "I-ORG",
        "U-DATE": "B-TIM",
        "B-DATE": "B-TIM",
        "I-DATE": "I-TIM",
        "L-DATE": "I-TIM"
    }

    for tag in tags:
        if tag in tag_map.keys():
            mapped_tags.append(tag_map[tag])
        else:
            mapped_tags.append("O")

    return mapped_tags

def read_json_data(file_path: str):
    sentences = []
    count = 0
    for data in read_jsonl(file_path):
        text, tags = data
        text = clean_text(text)
        
        words = word_tokenizer(text)
        tags = map_tags(tags)

        if len(words) != len(tags):
            count += 1
        else:
            sentences.append([[word, tag] for word, tag in zip(words, tags)])

    print(count)
    return sentences


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()