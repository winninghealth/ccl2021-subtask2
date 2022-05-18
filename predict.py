# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from overrides import overrides

from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance, Vocabulary

from trainer import build_model, init_logger
from data_loader import SymptomRecognitionDatasetReader, MAX_INPUT_LEN
from data_preprocess import SPECIALTOKENS, TENDS2IDX, ACTS2IDX, read_json_file

logger = logging.getLogger(__name__)

class SymptomRecognitionPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab
        
    def predict(self, tokens, token_type_ids, intends_ids, actions_ids, mentions, symptoms) -> JsonDict:
        response = self.predict_json({"tokens":tokens,
                                      "token_type_ids":token_type_ids,
                                      "intends_ids":intends_ids,
                                      "actions_ids":actions_ids,
                                      "mentions":mentions,
                                      "symptoms":symptoms})
        return response['tags']
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['tokens']
        token_type_ids = json_dict['token_type_ids']
        intends_ids = json_dict['intends_ids']
        actions_ids = json_dict['actions_ids']
        mentions = json_dict['mentions']
        symptoms = json_dict['symptoms']
        return self._dataset_reader.text_to_instance(tokens, token_type_ids, intends_ids, actions_ids, mentions, symptoms)

def read_input_file(input_path, dialogue_intention, dialogue_symptom, dialogue_diagnosis):
    test_file = read_json_file(input_path)
    test_symptom = read_json_file(dialogue_symptom)
    test_intention = read_json_file(dialogue_intention)
    test_diagnosis = read_json_file(dialogue_diagnosis)
    
    data = dict()
    for eid in test_file:
        if eid not in data:
            data[eid] = dict()
        data[eid]['implicit_info'] = test_symptom[eid]['implicit_info']
        for sid in test_file[eid]['dialogue']:
            sentence_id = sid['sentence_id']
            data[eid][sentence_id] = dict()
            
            sentence = sid['sentence']
            data[eid][sentence_id]['sentence'] = sentence
            
            speaker = sid['speaker']
            data[eid][sentence_id]['speaker'] = speaker
            
            diagnosis = test_diagnosis[eid]
            data[eid][sentence_id]['diagnosis'] = diagnosis
            
            mention = test_symptom[eid][sentence_id]
            data[eid][sentence_id]['sym_norm'] = mention
            
            dialogue_ir = test_intention[eid][sentence_id]
            if dialogue_ir != 'Diagnose' and dialogue_ir != 'Other':
                dialogue_tend, dialogue_act = dialogue_ir.split('-')
            else:
                dialogue_tend = dialogue_ir
                dialogue_act = dialogue_ir
            data[eid][sentence_id]['intention'] = dialogue_tend
            data[eid][sentence_id]['action'] = dialogue_act
            
    output_data = dict()
    for eid in data:
        output_data[eid] = dict()
        output_data[eid]['label'] = dict()
        output_data[eid]['doctor_round'] = []
        output_data[eid]['intends_dict'] = dict()
        output_data[eid]['actions_dict'] = dict()
        
        dialogue = []
        for sid in data[eid]:
            # Exclude the dialogue sentence that dialogue_act is Other and no special entity is mentioned
            if sid == 'implicit_info' or (data[eid][sid]['intention'] == 'Other' and data[eid][sid]['action'] == 'Other' and data[eid][sid]['sym_norm'] == []):
                continue
            sentence = list(data[eid][sid]['sentence'])
            speaker = [SPECIALTOKENS[data[eid][sid]['speaker']]]
            intends = TENDS2IDX[data[eid][sid]['intention']]
            acts = ACTS2IDX[data[eid][sid]['action']]
            dialogue_tend = [SPECIALTOKENS[data[eid][sid]['intention']]]
            dialogue_act = [SPECIALTOKENS[data[eid][sid]['intention']]]
            diagnosis = [SPECIALTOKENS[data[eid][sid]['diagnosis']]]
            mentions = data[eid][sid]['sym_norm']
            
            for i in mentions:
                i_type = i['type']
                i_sidx = i['start'] + len(dialogue) + 2 + len(dialogue_tend + dialogue_act)
                i_eidx = i['end'] + len(dialogue) + 2 + len(dialogue_tend + dialogue_act)
                if i_type not in output_data[eid]['label']:
                    output_data[eid]['label'][i_type] = dict()
                    output_data[eid]['label'][i_type]['mention'] = []
                output_data[eid]['label'][i_type]['mention'].append([i_sidx, i_eidx])
            
            current_length = len(dialogue)
            dialogue += diagnosis + speaker + dialogue_tend + dialogue_act + sentence
            
            if data[eid][sid]['speaker'] == '医生':
                output_data[eid]['doctor_round'].append([current_length, len(dialogue)])
            if intends not in output_data[eid]['intends_dict']:
                output_data[eid]['intends_dict'][intends] = []
            output_data[eid]['intends_dict'][intends].append([current_length, len(dialogue)])
            if acts not in output_data[eid]['actions_dict']:
                output_data[eid]['actions_dict'][acts] = []
            output_data[eid]['actions_dict'][acts].append([current_length, len(dialogue)])
            
        output_data[eid]['dialogue'] = ' '.join(dialogue)
    return output_data

def predict(pred_config):
    device = torch.device(pred_config.cuda_id if torch.cuda.is_available() else "cpu")
    
    serialization_dir = os.path.join(pred_config.model_dir)
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    vocab = Vocabulary.from_files(vocabulary_dir)
    
    dataset_reader = SymptomRecognitionDatasetReader(transformer_load_path=pred_config.pretrained_model_dir,
                                                     symptom_norm_file=pred_config.symptom_norm_file)
    
    model_dir = os.path.join(serialization_dir, pred_config.model_name)
    model = build_model(vocab, pred_config.pretrained_model_dir, len(dataset_reader.SymNorm2idx))
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    
    predictor = SymptomRecognitionPredictor(model, dataset_reader)
    
    data_file = read_input_file(pred_config.test_input_file, 
                                pred_config.dialogue_intention, 
                                pred_config.dialogue_symptom, 
                                pred_config.dialogue_diagnosis)
    
    predict_result = dict()
    for eid in tqdm(data_file.keys()):
        predict_result[eid] = dict()
        tokens = data_file[eid]['dialogue'].split()
        token_type_ids  = [0] * len(tokens)
        intends_ids = [TENDS2IDX['Other']] * len(tokens)
        actions_ids = [ACTS2IDX['Other']] * len(tokens)
        
        # input_ids
        tokens = dataset_reader._split_dialogue(tokens, MAX_INPUT_LEN-2)
        
        # simply regard token_type_ids as speaker ids here
        for i in data_file[eid]['doctor_round']:
            for j in range(i[0],i[1]):
                token_type_ids[j] = 1
        token_type_ids = dataset_reader._split_dialogue(token_type_ids, MAX_INPUT_LEN-2)
        
        # intention_type_ids
        for i in data_file[eid]['intends_dict']:
            for j in data_file[eid]['intends_dict'][i]:
                for k in range(j[0],j[1]):
                    intends_ids[k] = int(i)
        intends_ids = dataset_reader._split_dialogue(intends_ids, MAX_INPUT_LEN-2)
        
        # action_type_ids
        for i in data_file[eid]['actions_dict']:
            for j in data_file[eid]['actions_dict'][i]:
                for k in range(j[0],j[1]):
                    actions_ids[k] = int(i)
        actions_ids = dataset_reader._split_dialogue(actions_ids, MAX_INPUT_LEN-2)
        
        symptoms = []
        mentions = dict()
        for sym_ment in data_file[eid]['label'].keys():
            symptoms.append(dataset_reader.SymNorm2idx[sym_ment])
            mentions[dataset_reader.SymNorm2idx[sym_ment]] = data_file[eid]['label'][sym_ment]['mention']
        result = predictor.predict(tokens, token_type_ids, intends_ids, actions_ids, mentions, symptoms)
        for idx, i in enumerate(symptoms):
            predict_result[eid][dataset_reader.idx2SymNorm[i]] = str(result[idx])
    
    pred_path = os.path.join(pred_config.test_output_file)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(predict_result, json_file, ensure_ascii=False, indent=4)
    
    logger.info("Prediction Done!")

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_input_file", default="./data/dataset/IMCS_test.json", type=str)
    parser.add_argument("--test_output_file", default="IMCS-SR_test.json", type=str)
    parser.add_argument("--symptom_norm_file", default='./data/dataset/symptom_norm.csv', type=str)
    
    parser.add_argument("--model_dir", default="./save_model", type=str)
    parser.add_argument("--model_name", default="best.th", type=str)
    parser.add_argument("--pretrained_model_dir", default="./plms/roberta_base")
    
    parser.add_argument("--dialogue_diagnosis", default="./data/imcs_results/IMCS-DIAG_test.json", type=str)
    parser.add_argument("--dialogue_intention", default="./data/imcs_results/IMCS-IR_test.json", type=str)
    parser.add_argument("--dialogue_symptom", default="./data/imcs_results/IMCS-NORM_test.json", type=str)
    
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    
    pred_config = parser.parse_args()
    predict(pred_config)
    