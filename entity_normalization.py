# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

from data_preprocess import read_json_file, save_json_file

'''
从训练集和验证集中生成归一化字典
'''
def create_normalized_dict(data):
    norm2mention = dict()
    mention2norm = dict()
    for eid in data.keys():
        sentence = data[eid]['dialogue'].split()
        for n in data[eid]['label']:
            if n not in norm2mention:
                norm2mention[n] = dict()
            for m in data[eid]['label'][n]['mention']:
                mention = ''.join(sentence[m[0]:m[1]])
                if mention not in norm2mention[n]:
                    norm2mention[n][mention] = 0
                norm2mention[n][mention] += 1
                if mention not in mention2norm:
                    mention2norm[mention] = list()
                if n not in mention2norm[mention]:
                    mention2norm[mention].append(n)
    return norm2mention, mention2norm

'''
统计并剔除有歧义的提及，歧义是指该提及对应多于一种的标准词
'''
def return_ambiguous_mention(mention2norm):
    ambiguous_term = dict()
    for i in mention2norm:
        if len(mention2norm[i]) > 1:
            ambiguous_term[i] = mention2norm[i]
    print('以下提及存在歧义：')
    for i in ambiguous_term:
        print(i, ambiguous_term[i])
        del mention2norm[i]
    return ambiguous_term, mention2norm

'''
test_file: 读取 IMCS_test.json 的文件
test_ner_file: 读取提交给 IMCS-NER 任务的 IMCS-NER_test.json 文件
读取BIO格式的症状实体（待归一化），转换成span格式
'''
def bio2span(test_file, test_ner_file):
    span_info = []
    assert_num = 0
    for eid in test_file:
        for sid in test_file[eid]['dialogue']:
            sentence_id = sid['sentence_id']
            sentence = sid['sentence']
            bio_label = test_ner_file[eid][sentence_id].split()
            assert len(sentence) == len(bio_label)
            assert_num += bio_label.count('B-Symptom')
            
            start_index, end_index = 0, 0
            entity_name, entity_type = str(), str()
            for idx, i in enumerate(bio_label):
                bio = i
                if bio == 'O' and entity_name != '':
                    end_index = idx
                    if entity_type == 'Symptom':
                        span_info.append((eid, sentence_id, start_index, end_index, entity_name))
                    entity_name, entity_type = str(), str()
                if bio[0] == 'B':
                    if entity_name != '':
                        end_index = idx
                        if entity_type == 'Symptom':
                            span_info.append((eid, sentence_id, start_index, end_index, entity_name))
                        entity_name, entity_type = str(), str()
                        start_index = idx
                        entity_type = bio[2:]
                        entity_name += sentence[idx]
                    else:
                        start_index = idx
                        entity_type = bio[2:]
                        entity_name += sentence[idx]
                if bio[0] == 'I':
                    entity_name += sentence[idx]
            
            if entity_name != '':
                end_index = len(bio_label)
                if entity_type == 'Symptom':
                    span_info.append((eid, sentence_id, start_index, end_index, entity_name))
    assert len(span_info) == assert_num
    print('测试集中有{}个实体'.format(assert_num))
    unique_span = [i[-1] for i in span_info]
    unique_span = set(unique_span)
    print('测试集中有{}种实体提及'.format(len(unique_span)))
    return span_info

'''
统计在测试集中，无法通过标准归一化字典进行归一化的提及
'''
def unknown_mention(test_span_info, mention2norm):
    unknown_m = dict()
    for i in test_span_info:
        mention = i[-1]
        if mention not in mention2norm:
            if mention not in unknown_m:
                unknown_m[mention] = 0
            unknown_m[mention] += 1
    unknown_freq = [unknown_m[i] for i in unknown_m]
    print('测试集中有{}种实体无法完成归一化，共出现了{}次'.format(len(unknown_m), sum(unknown_freq)))
    return unknown_m

'''
生成测试集的归一化结果：test_dialogue_symptoms.json 文件
'''
def generate_normalization_file(test_data, test_span_info, mention2norm, output_path):
    unnorm = 0
    output_dict = dict()
    for eid in test_data.keys():
        output_dict[eid] = dict()
        output_dict[eid]['implicit_info'] = list()
        for sid in test_data[eid]['dialogue']:
            sentence_id = sid['sentence_id']
            output_dict[eid][sentence_id] = []
    for i in test_span_info:
        eid, sid, start_idx, end_idx, mention = i
        if mention in mention2norm:
            norm = mention2norm[mention][0]
        else:
            '''
            通过规则与TF-IDF等算法实现归一化，此部分暂不提供
            '''
            unnorm += 1
            continue
        output_dict[eid][sid].append({'name':mention,
                                      'type':norm,
                                      'start':start_idx,
                                      'end':end_idx})
        if norm not in output_dict[eid]['implicit_info']:
            output_dict[eid]['implicit_info'].append(norm)
    save_json_file(output_path, output_dict)
    print('有{}个实体未完成归一化'.format(unnorm))


if __name__ == "__main__":
    
    train_path = './data/train_corpus.json'
    dev_path = './data/dev_corpus.json'
    
    test_path = './data/dataset/IMCS_test.json'
    test_submission_path = './data/imcs_results/IMCS-NER_test.json'
    
    train_data = read_json_file(train_path)
    dev_data = read_json_file(dev_path)
    test_data = read_json_file(test_path)
    test_result = read_json_file(test_submission_path)
    
    data = dict()
    data.update(train_data)
    data.update(dev_data)
    
    norm2mention, mention2norm = create_normalized_dict(data)
    ambiguous_term, mention2norm = return_ambiguous_mention(mention2norm)
    print('----------------------')
    test_span_info = bio2span(test_data, test_result)
    print('----------------------')
    unknown_m = unknown_mention(test_span_info, mention2norm)
    print('----------------------')
    output_path = './data/imcs_results/IMCS-NORM_test.json'
    generate_normalization_file(test_data, test_span_info, mention2norm, output_path)
    