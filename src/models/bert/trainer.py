from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import AdamW, get_scheduler
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pickle
import os
import torch
from bert import data_loader

def count_tags(dataset):
    tags = {}
    for sentence in dataset:
        for word, tag in sentence:
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1
    return tags

def train_csebuet_bert_model(data_path, model_path):

    with open(os.path.join(data_path, "train", "train.pkl"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, "test", "test.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    all_data = train_data + test_data
    tag_counts = count_tags(all_data)

    entity_mapping = {key: i for i, key in enumerate(tag_counts.keys())}
    label_mapping = {value: key for key, value in entity_mapping.items()}
    with open("models/label_mapping.pkl", 'wb') as f:
        pickle.dump(label_mapping, f)


    model_checkpoint = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(entity_mapping),
        output_attentions = False,
        output_hidden_states = False)
    
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-12
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    MAX_LEN = 128

    data_train = data_loader.NERDataset(train_data, tokenizer, entity_mapping, MAX_LEN)
    data_val = data_loader.NERDataset(test_data, tokenizer, entity_mapping, MAX_LEN)

    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=32, num_workers=4
    )

    loader_val = torch.utils.data.DataLoader(
        data_val, batch_size=32, num_workers=4
    )

    # add scheduler to linearly reduce the learning rate throughout the epochs.
    num_epochs = 3
    num_training_steps = num_epochs * len(loader_train)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        model.train()
        final_loss = 0
        predictions , true_labels = [], []
        for batch in loader_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            true_labels.extend(batch['labels'].detach().cpu().numpy().ravel())
            predictions.extend(np.argmax(outputs[1].detach().cpu().numpy(), axis=2).ravel())
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            final_loss+=loss.item()
            
        print(f'Training loss: {final_loss/len(loader_train)}')
        print('Training F1: {}'.format(f1_score(predictions, true_labels, average='macro')))
        print(f'Training acc: {accuracy_score(predictions, true_labels)}')
        print('*'*20)
        
        model.eval()
        final_loss = 0
        predictions , true_labels = [], []
        for batch in loader_val:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            final_loss+=outputs.loss.item()
            true_labels.extend(batch['labels'].detach().cpu().numpy().ravel())
            predictions.extend(np.argmax(outputs[1].detach().cpu().numpy(), axis=2).ravel())
        print(f'Validation loss: {final_loss/len(loader_val)}')
        print('Vallidation F1: {}'.format(f1_score(predictions, true_labels, average='macro')))
        print(f'Validaton acc: {accuracy_score(predictions, true_labels)}')
        print('*'*20)
        torch.save(model, os.path.join(model_path, "bn_ner_csebuet_bert.pt"))