# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import json
import torch
import logging
from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TensorField, MetadataField, ListField

from transformers import BertTokenizer
from data_preprocess import TENDS2IDX, ACTS2IDX

logger = logging.getLogger(__name__)

MAX_INPUT_LEN = 512

class SymptomRecognitionDatasetReader(DatasetReader):
    def __init__(self, transformer_load_path:str, symptom_norm_file:str, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(transformer_load_path)
        attr_to_idx = dict()
        idx_to_attr = dict()
        with open(symptom_norm_file, 'r', encoding='utf-8') as f:
            content = f.readlines()
            content = [i.strip() for i in content if i != 'norm']
            for idx, sym in enumerate(content):
                attr_to_idx[sym] = idx
                idx_to_attr[idx] = sym
        self.SymNorm2idx = attr_to_idx
        self.idx2SymNorm = idx_to_attr
    
    def _split_dialogue(self, tokens, split_len):
        res_data = []
        if len(tokens) > split_len:
            for i in range(int(len(tokens)/split_len)):
                split_tokens = tokens[split_len*i:split_len*(i + 1)]
                res_data.append(split_tokens)
            last_tokens = tokens[int(len(tokens)/split_len)*split_len:]
            if last_tokens:
                res_data.append(last_tokens)
        else:
            res_data.append(tokens)
        return res_data
    
    @overrides
    def _read(self, file_path):        
        with open(file_path, "r", encoding='utf-8') as file:
            data_file = json.load(file)
            for eid in data_file.keys():
                tokens = data_file[eid]['dialogue'].split()
                token_type_ids  = [0] * len(tokens)
                intends_ids = [TENDS2IDX['Other']] * len(tokens)
                actions_ids = [ACTS2IDX['Other']] * len(tokens)
                
                # input_ids
                tokens = self._split_dialogue(tokens, MAX_INPUT_LEN-2)
                
                # simply regard token_type_ids as speaker ids here
                for i in data_file[eid]['doctor_round']:
                    for j in range(i[0], i[1]):
                        token_type_ids[j] = 1
                token_type_ids = self._split_dialogue(token_type_ids, MAX_INPUT_LEN-2)
                
                # intention_type_ids
                for i in data_file[eid]['intends_dict']:
                    for j in data_file[eid]['intends_dict'][i]:
                        for k in range(j[0], j[1]):
                            intends_ids[k] = int(i)
                intends_ids = self._split_dialogue(intends_ids, MAX_INPUT_LEN-2)
                
                # action_type_ids
                for i in data_file[eid]['actions_dict']:
                    for j in data_file[eid]['actions_dict'][i]:
                        for k in range(j[0], j[1]):
                            actions_ids[k] = int(i)
                actions_ids = self._split_dialogue(actions_ids, MAX_INPUT_LEN-2)
                
                # golden label
                labels = []
                symptoms = []
                mentions = dict()
                for sym_ment in data_file[eid]['label'].keys():
                    symptoms.append(self.SymNorm2idx[sym_ment])
                    labels.append(int(data_file[eid]['label'][sym_ment]['type']))
                    mentions[self.SymNorm2idx[sym_ment]] = data_file[eid]['label'][sym_ment]['mention']
                if mentions == dict():
                    continue
                yield self.text_to_instance(tokens, token_type_ids, intends_ids, actions_ids, mentions, symptoms, labels)
    
    def text_to_instance(self, tokens, token_type_ids, intends_ids, actions_ids, mentions, symptoms, labels = None) -> Instance:
        fields: Dict[str, Field] = {}
        
        tokens = [['[CLS]'] + token + ['[SEP]'] for token in tokens]
        tokens_field = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        tokens_field = [TensorField(torch.tensor(t)) for t in tokens_field]
        fields["tokens"] = ListField(tokens_field)
        
        token_type_ids = [TensorField(torch.tensor(t)) for t in token_type_ids]
        fields["token_type"] = ListField(token_type_ids)
        
        intends_ids = [TensorField(torch.tensor(t)) for t in intends_ids]
        fields["intends_ids"] = ListField(intends_ids)
        
        actions_ids = [TensorField(torch.tensor(t)) for t in actions_ids]
        fields["actions_ids"] = ListField(actions_ids)
        
        fields["mentions"] = MetadataField(mentions)
        fields["symptoms"] = TensorField(torch.tensor(symptoms).long())
        
        if labels is not None:
            fields["labels"] = TensorField(torch.tensor(labels))
        
        return Instance(fields)
