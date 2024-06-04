"""Utility functions for creating embedding features by pre-trained language model."""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import get_min_max_scores


def normalize_scores(y, essay_set, attribute_name) -> list:
    """
    Normalize scores based on the min and max scores for each unique prompt_id in essay_set.
    Args:
        y: Scores to normalize.
        essay_set: Array of essay_set (prompt_id) for each score.
        attribute_name: The attribute name to filter the min and max scores.
    Returns:
        np.ndarray: Normalized scores.
    """
    min_max_scores = get_min_max_scores()
    normalized_scores = np.zeros_like(y, dtype=float)
    for unique_prompt_id in np.unique(essay_set):
        minscore, maxscore = min_max_scores[unique_prompt_id][attribute_name]
        mask = (essay_set == unique_prompt_id)
        normalized_scores[mask] = (y[mask] - minscore) / (maxscore - minscore)
    return normalized_scores.tolist()


def load_data(data_path: str, attribute: str = 'score') -> dict:
    """
    Load data from the given path.
    Args:
        data_path: Path to the data.
        attribute: The attribute to read.
    Returns:
        dict: Data.
    """
    data = {}
    for file in ['train', 'dev', 'test']:
        feature = []
        label = []
        essay_id = []
        essay_set = []
        try:
            read_data = pd.read_pickle(data_path + file + '.pkl')
        except:
            read_data = pd.read_pickle(data_path + file + '.pk')
        for i in range(len(read_data)):
            feature.append(read_data[i]['content_text'])
            label.append(int(read_data[i][attribute]))
            essay_id.append(int(read_data[i]['essay_id']))
            essay_set.append(int(read_data[i]['prompt_id']))
        data[file] = {'feature': feature, 'label': label, 'essay_id': essay_id, 'essay_set': essay_set}

    return data


class EssayDataset(Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Args:
            data: Data.
            tokenizer: Tokenizer.
            max_length: Maximum length of the input.
        """
        self.texts = np.array(data['feature'])
        self.scores = np.array(data['normalized_label'])
        self.prompts = np.array(data['essay_set'])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'score': torch.tensor(self.scores[item], dtype=torch.float),
            'prompt': torch.tensor(self.prompts[item], dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


class ASAP_Dataloader():
    def __init__(self, data_path, tokenizer, attribute='score', batch_size=32, val_batch_size=30, max_length=512, num_val_samples=30):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.attribute = attribute
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.max_length = max_length
        self.num_val_samples = num_val_samples

        self.data = load_data(data_path, self.attribute)
        self.train_data = self.data['train']
        self.dev_data = self.data['dev']
        self.target_data = self.data['test']
        self.train_data['normalized_label'] = normalize_scores(np.array(self.train_data['label']), np.array(self.train_data['essay_set']), attribute)
        self.dev_data['normalized_label'] = normalize_scores(np.array(self.dev_data['label']), np.array(self.dev_data['essay_set']), attribute)
        self.target_data['normalized_label'] = normalize_scores(np.array(self.target_data['label']), np.array(self.target_data['essay_set']), attribute)

        # Combine train and dev data to create source data
        self.source_data = {
            'feature': self.train_data['feature'] + self.dev_data['feature'],
            'label': self.train_data['label'] + self.dev_data['label'],
            'essay_id': self.train_data['essay_id'] + self.dev_data['essay_id'],
            'essay_set': self.train_data['essay_set'] + self.dev_data['essay_set'],
            'normalized_label': self.train_data['normalized_label'] + self.dev_data['normalized_label']
        }

        # Create validation data from target data and remove selected data from target
        # Select indices to ensure a diverse range of scores in the validation set
        labels_array = np.array(self.target_data['label'])
        # Calculate quantiles to ensure a spread of scores
        quantiles = np.quantile(labels_array, np.linspace(0, 1, self.num_val_samples))
        val_indices = []
        for q in quantiles:
            # Find the index of the closest value to each quantile
            idx = (np.abs(labels_array - q)).argmin()
            if idx not in val_indices:  # Ensure unique indices
                val_indices.append(idx)
        
        self.val_data = {
            'feature': [],
            'label': [],
            'essay_id': [],
            'essay_set': [],
            'normalized_label': []
        }
        
        for i in sorted(val_indices, reverse=True):
            self.val_data['feature'].append(self.target_data['feature'].pop(i))
            self.val_data['label'].append(self.target_data['label'].pop(i))
            self.val_data['essay_id'].append(self.target_data['essay_id'].pop(i))
            self.val_data['essay_set'].append(self.target_data['essay_set'].pop(i))
            self.val_data['normalized_label'].append(self.target_data['normalized_label'].pop(i))

        self.source_dataset = EssayDataset(self.source_data, self.tokenizer, self.max_length)
        self.val_dataset = EssayDataset(self.val_data, self.tokenizer, self.max_length)
        self.target_dataset = EssayDataset(self.target_data, self.tokenizer, self.max_length)

    def run(self):
        source_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=0, shuffle=False, drop_last=True)
        target_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False, drop_last=False)

        return source_loader, val_loader, target_loader

