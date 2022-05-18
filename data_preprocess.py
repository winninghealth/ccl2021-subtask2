# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import json

SPECIALTOKENS = {# 2 Speaker
                 '患者':'[unused1]',
                 '医生':'[unused2]',
                 # 6 Diagnosis
                 '小儿消化不良':'[unused3]',
                 '上呼吸道感染':'[unused4]',
                 '小儿支气管炎':'[unused5]',
                 '小儿腹泻':'[unused6]',
                 '小儿发热':'[unused7]',
                 '小儿感冒':'[unused8]',
                 # 2(+2) Intentions
                 'Request':'[unused9]',
                 'Inform':'[unused10]',
                 # 7(+2) Actions
                 'Symptom':'[unused11]',
                 'Etiology':'[unused12]',
                 'Basic_Information':'[unused13]',
                 'Existing_Examination_and_Treatment':'[unused14]',
                 'Drug_Recommendation':'[unused15]',
                 'Medical_Advice':'[unused16]',
                 'Precautions':'[unused17]',
                 # Shared by Intentions and Actions
                 'Diagnose':'[unused18]',
                 'Other':'[unused19]'}

# 4(2+2) Intentions
TENDS2IDX = {'Request':0,
             'Inform':1,
             'Diagnose':2,
             'Other':3}

# 9(7+2) Actions
ACTS2IDX = {'Symptom':0,
            'Etiology':1,
            'Basic_Information':2,
            'Existing_Examination_and_Treatment':3,
            'Drug_Recommendation':4,
            'Medical_Advice':5,
            'Precautions':6,
            'Diagnose':7,
            'Other':8}

def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def save_json_file(file, output_data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def process_bio2span(data):
    train_span = dict()
    for eid in data:
        train_span[eid] = dict()
        train_span[eid]['implicit_info'] = data[eid]['implicit_info']['Symptom']
        for sid in data[eid]['dialogue']:
            train_span[eid][sid['sentence_id']] = dict()
            train_span[eid][sid['sentence_id']]['sentence'] = sid['sentence']
            train_span[eid][sid['sentence_id']]['speaker'] = sid['speaker']
            train_span[eid][sid['sentence_id']]['dialogue_act'] = sid['dialogue_act']
            train_span[eid][sid['sentence_id']]['BIO_label'] = sid['BIO_label']
            train_span[eid][sid['sentence_id']]['diagnosis'] = data[eid]['diagnosis']
            
            sentence = list(sid['sentence'])
            biolabel = sid['BIO_label'].split()
            assert len(sentence) == len(biolabel)
            
            span_info, sym_info = list(), list()
            start_index, end_index = 0, 0
            entity_name, entity_type = str(), str()
            
            symptom_idx = 0
            symptom_norm = sid['symptom_norm']
            
            for idx, i in enumerate(zip(sentence, biolabel)):
                char, bio = i[0], i[1]
                if bio == 'O' and entity_name != '':
                    end_index = idx
                    span_info.append({'name':entity_name,
                                      'type':entity_type,
                                      'start':start_index,
                                      'end':end_index})
                    if entity_type == 'Symptom':
                        sym_info.append({'name':entity_name,
                                         'type':symptom_norm[symptom_idx],
                                         'start':start_index,
                                         'end':end_index})
                        symptom_idx += 1
                    assert entity_name == ''.join(sentence[start_index:end_index])
                    t_list = list()
                    for t in biolabel[start_index:end_index]:
                        t_list.append(t[2:])
                    assert len(set(t_list)) == 1
                    assert t_list[0] == entity_type
                    entity_name, entity_type = str(), str()
                if bio[0] == 'B':
                    if entity_name != '':
                        end_index = idx
                        span_info.append({'name':entity_name,
                                          'type':entity_type,
                                          'start':start_index,
                                          'end':end_index})
                        if entity_type == 'Symptom':
                            sym_info.append({'name':entity_name,
                                             'type':symptom_norm[symptom_idx],
                                             'start':start_index,
                                             'end':end_index})
                            symptom_idx+=1
                        assert entity_name == ''.join(sentence[start_index:end_index])
                        t_list = list()
                        for t in biolabel[start_index:end_index]:
                            t_list.append(t[2:])
                        assert len(set(t_list)) == 1
                        assert t_list[0] == entity_type
                        entity_name, entity_type = str(), str()
                        start_index = idx
                        entity_type = bio[2:]
                        entity_name += char
                    else:
                        start_index = idx
                        entity_type = bio[2:]
                        entity_name += char
                if bio[0] == 'I':
                    entity_name += char
            
            if entity_name != '':
                end_index = len(sentence)
                span_info.append({'name':entity_name,
                                  'type':entity_type,
                                  'start':start_index,
                                  'end':end_index})
                if entity_type == 'Symptom':
                    sym_info.append({'name':entity_name,
                                     'type':symptom_norm[symptom_idx],
                                     'start':start_index,
                                     'end':end_index})
                    symptom_idx+=1
                assert entity_name == ''.join(sentence[start_index:end_index])
                t_list = list()
                for t in biolabel[start_index:end_index]:
                    t_list.append(t[2:])
                assert len(set(t_list)) == 1
                assert t_list[0] == entity_type
            assert len(symptom_norm) == symptom_idx
            
            train_span[eid][sid['sentence_id']]['entities'] = span_info
            train_span[eid][sid['sentence_id']]['sym_norm'] = sym_info
    return train_span

def preprocess_data(data):
    output_data = dict()
    for eid in data:
        output_data[eid] = dict()
        output_data[eid]['label'] = dict()
        # Only record one speaker here
        output_data[eid]['doctor_round'] = []
        output_data[eid]['intends_dict'] = dict()
        output_data[eid]['actions_dict'] = dict()
        
        dialogue = []
        for sid in data[eid]:
            # Exclude the dialogue sentence that dialogue_act is Other and no special entity is mentioned
            if sid == 'implicit_info' or (data[eid][sid]['dialogue_act'] == 'Other' and set(data[eid][sid]['BIO_label'].split()) == {'O'}):
                continue
            sentence = list(data[eid][sid]['sentence'])
            speaker = [SPECIALTOKENS[data[eid][sid]['speaker']]]
            if data[eid][sid]['dialogue_act'] != 'Diagnose' and data[eid][sid]['dialogue_act'] != 'Other':
                dialogue_tend, dialogue_act = data[eid][sid]['dialogue_act'].split('-')
                intends = TENDS2IDX[dialogue_tend]
                acts = ACTS2IDX[dialogue_act]
                dialogue_tend = [SPECIALTOKENS[dialogue_tend]]
                dialogue_act = [SPECIALTOKENS[dialogue_act]]
            else:
                dialogue_tend = []
                dialogue_act = [SPECIALTOKENS[data[eid][sid]['dialogue_act']]]
                intends = TENDS2IDX[data[eid][sid]['dialogue_act']]
                acts = ACTS2IDX[data[eid][sid]['dialogue_act']]    
            diagnosis = [SPECIALTOKENS[data[eid][sid]['diagnosis']]]
            
            for i in data[eid][sid]['sym_norm']:
                i_type = i['type']
                i_sidx = i['start'] + len(dialogue) + 2 + len(dialogue_tend + dialogue_act)
                i_eidx = i['end'] + len(dialogue) + 2 + len(dialogue_tend + dialogue_act)
                if i_type not in output_data[eid]['label']:
                    output_data[eid]['label'][i_type] = dict()
                    output_data[eid]['label'][i_type]['type'] = data[eid]['implicit_info'][i_type]
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

def process_file(data_path, data_save_path):
    data = read_json_file(data_path)
    data_span = process_bio2span(data)
    data = preprocess_data(data_span)
    save_json_file(data_save_path, data)

if __name__ == "__main__":
    
    train_path = './data/dataset/IMCS_train.json'
    train_save_path = './data/train_corpus.json'
    process_file(train_path, train_save_path)
    
    dev_path = './data/dataset/IMCS_dev.json'
    dev_save_path = './data/dev_corpus.json'
    process_file(dev_path, dev_save_path)
    