# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import torch
import random
from overrides import overrides
from typing import Dict, Optional
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import FBetaMeasure

from modeling_gats import GAT
from transformers import BertModel
from data_preprocess import TENDS2IDX, ACTS2IDX

class SymptomRecognitionModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        span_rep: str,
        symptom_norm_num: int,
        lstmencoder: Seq2SeqEncoder,
        transformer_load_path: str,
        dropout: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.span_rep = span_rep
        assert self.span_rep in ['Mean','Concat']
        
        self.lstmencoder = lstmencoder
        self.bert_encoder = BertModel.from_pretrained(transformer_load_path)
        self.sym_dense = torch.nn.Linear(in_features=1024, out_features=3)
        if self.span_rep == 'Concat':
            self.gats = GAT(n_feat=self.lstmencoder.get_output_dim()*2, n_hid=128, dropout=0.4, alpha=0.2, n_heads=8)
        elif self.span_rep == 'Mean':
            self.gats = GAT(n_feat=self.lstmencoder.get_output_dim(), n_hid=128, dropout=0.4, alpha=0.2, n_heads=8)
        # Embeddings
        self.symptom_embedder = torch.nn.Embedding(num_embeddings=symptom_norm_num,
                                                   embedding_dim=self.lstmencoder.get_output_dim()*2)
        self.speaker_embedder = torch.nn.Embedding(num_embeddings=2, embedding_dim=32)
        self.intends_embedder = torch.nn.Embedding(num_embeddings=len(TENDS2IDX), embedding_dim=32)
        self.actions_embedder = torch.nn.Embedding(num_embeddings=len(ACTS2IDX), embedding_dim=32)
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.calculate_f1 = {
            "F1-micro": FBetaMeasure(average='micro'),
            "F1-class": FBetaMeasure(average=None, labels=[0,1,2])
        }
        self.loss = torch.nn.CrossEntropyLoss()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        initializer(self)
    
    @overrides
    def forward(self, tokens, token_type, intends_ids, actions_ids, mentions, symptoms, labels = None, **kwargs):
        
        tokens_mask = tokens != 0
        batch_size, split_len, seq_len = tokens.shape
        tokens = tokens.reshape(batch_size*split_len, seq_len)
        
        token_type = token_type.reshape(batch_size, -1)
        token_type = self.speaker_embedder(token_type)
        
        intends_ids = intends_ids.reshape(batch_size, -1)
        intends_ids = self.intends_embedder(intends_ids)
        
        actions_ids = actions_ids.reshape(batch_size, -1)
        actions_ids = self.actions_embedder(actions_ids)
        
        # bert encoder feature
        hidden_state = self.bert_encoder(input_ids=tokens, return_dict=True)['last_hidden_state']
        hidden_state = hidden_state.reshape(batch_size, split_len, seq_len, -1)
        hidden_state = hidden_state[:,:,1:-1,:]
        hidden_state = hidden_state.reshape(batch_size, split_len*(seq_len-2), -1)
        if self.dropout:
            hidden_state = self.dropout(hidden_state)
        tokens_mask = tokens_mask[:,:,1:-1]
        tokens_mask = tokens_mask.reshape(batch_size, split_len * (seq_len - 2))
        
        hidden_state = torch.cat((hidden_state, token_type), dim=-1)
        hidden_state = torch.cat((hidden_state, intends_ids), dim=-1)
        hidden_state = torch.cat((hidden_state, actions_ids), dim=-1)
        
        # lstm encoder feature
        encoded_text = self.lstmencoder(hidden_state, tokens_mask)
        # encoded_text: [batch_size, seq_len, hidden_size]
        if self.dropout:
            encoded_text = self.dropout(encoded_text)
        
        # nodes feature
        witness_size = 20
        witness_dict = dict()
        # embedded_sym_node: [batch_size, number_of_symptoms, embedding_size]
        embedded_sym_node = self.symptom_embedder(symptoms)
        if self.dropout:
            embedded_sym_node = self.dropout(embedded_sym_node)
        batch_size, symptoms_num, sym_hidden = embedded_sym_node.shape
        # span_features: [batch_size, number_of_symptoms, number_of_witness, embedding_size]
        span_features = torch.zeros(batch_size, symptoms_num, witness_size+1, sym_hidden).to(tokens.device)
        for i in range(batch_size):
            for j in range(symptoms_num):
                span_features[i][j][0] = embedded_sym_node[i][j]
                if symptoms[i][j].item() == 0:
                    witness_dict[(i,j)] = []
                else:
                    witness_dict[(i,j)] = mentions[i][symptoms[i][j].item()]
                if len(witness_dict[(i,j)]) > witness_size:
                    witness_dict[(i,j)] = random.sample(witness_dict[(i,j)], witness_size)
                for idx, k in enumerate(witness_dict[(i,j)]):
                    if self.span_rep == 'Mean':
                        span_features[i][j][idx+1] = torch.mean(encoded_text[i][k[0]:k[1]], dim=0)
                    if self.span_rep == 'Concat':
                        span_features[i][j][idx+1] = torch.cat((encoded_text[i][k[0]],encoded_text[i][k[1]-1]),dim=0)
        
        # adjacent matrix
        sym_mat = torch.zeros(witness_size+1, witness_size+1).to(tokens.device)
        for i in range(1, witness_size+1):
            sym_mat[i][0] = 1
        sym_mat = sym_mat.repeat(batch_size, symptoms_num, 1, 1)
        
        # graph attention
        span_features = span_features.reshape(-1, span_features.shape[2], span_features.shape[3])
        sym_mat = sym_mat.reshape(-1, sym_mat.shape[2], sym_mat.shape[3])
        
        embedded_sym_node = torch.zeros(batch_size*symptoms_num, 1024).to(tokens.device)
        for i in range(span_features.shape[0]):
            output_features = self.gats(span_features[i], sym_mat[i])
            embedded_sym_node[i] = output_features[0,:]
        
        embedded_sym_node = self.sym_dense(embedded_sym_node)
        embedded_sym_node = embedded_sym_node.reshape(batch_size, symptoms_num, -1)
        
        output = dict()
        # metric
        if labels != None:
            preds = embedded_sym_node
            labels_mask = symptoms != 0
            self.calculate_f1['F1-micro'](preds, labels, labels_mask)
            self.calculate_f1['F1-class'](preds, labels, labels_mask)
            
            preds_prob = self.logsoftmax(preds)
            preds_prob = torch.argmax(preds_prob, dim=-1)
            output['tags'] = preds_prob
        else:
            preds = embedded_sym_node
            preds_prob = self.logsoftmax(preds)
            preds_prob = torch.argmax(preds_prob, dim=-1)
            output['tags'] = preds_prob
        
        # loss
        if labels != None:
            embedded_sym_node = embedded_sym_node.reshape(-1, embedded_sym_node.shape[2])
            labels = labels.reshape(-1)
            output_loss = self.loss(embedded_sym_node, labels.long())
            output["loss"] = output_loss
            
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = dict()
        label_overall = self.calculate_f1['F1-micro'].get_metric(reset)
        label_class = self.calculate_f1['F1-class'].get_metric(reset)
        metrics_to_return['Micro-fscore'] = label_overall['fscore']
        
        for idx, lc in enumerate(['0','1','2']):
            metrics_to_return[lc+'-precision'] = label_class['precision'][idx]
            metrics_to_return[lc+'-recall'] = label_class['recall'][idx]
            metrics_to_return[lc+'-fscore'] = label_class['fscore'][idx]
        
        return metrics_to_return
