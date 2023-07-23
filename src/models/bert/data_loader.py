import torch

class NERDataset:
    def __init__(self, dataset, tokenizer, entity_mapping, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.entity_mapping = entity_mapping
        self.MAX_LEN = max_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        data = self.dataset[item]
        text = []
        tags = []
        for word, tag in data:
            text.append(word)
            tags.append(tag)
        
        ids = []
        target_tag =[]
        
        # tokenize words and define tags accordingly
        # running -> [run, ##ning]
        # tags - ['O', 'O']
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([self.entity_mapping[tags[i]]] * input_len)
        
        # truncate
        ids = ids[:self.MAX_LEN - 2]
        target_tag = target_tag[:self.MAX_LEN - 2]
        
        # add special tokenstag[:MAX_LEN - 2]
        
        # add special tokens
        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        
        # construct padding
        padding_len = self.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'labels': torch.tensor(target_tag, dtype=torch.long)
               }