# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
prepare_environment(Params({"random_seed":1000, "numpy_seed":2000, "pytorch_seed":3000}))

import os
import torch
import logging
import argparse

from allennlp.models.model import Model
from allennlp.data import DataLoader, Vocabulary
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from allennlp.training.learning_rate_schedulers.linear_with_warmup import LinearWithWarmup

from transformers.optimization import AdamW
from transformers import BertConfig
from modeling_sr import SymptomRecognitionModel
from data_loader import SymptomRecognitionDatasetReader

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def build_vocab(instances):
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary,
                transformer_load_path: str,
                symptom_norm_num: int,
                span_rep = 'Concat') -> Model:
    config = BertConfig.from_json_file(os.path.join(transformer_load_path, 'config.json'))
    lstmencoder = LstmSeq2SeqEncoder(input_size=config.hidden_size+32*3,
                                     hidden_size=128,
                                     num_layers=1,
                                     bidirectional=True)
    return SymptomRecognitionModel(vocab=vocab,
                                   span_rep=span_rep,
                                   symptom_norm_num=symptom_norm_num,
                                   lstmencoder=lstmencoder,
                                   transformer_load_path=transformer_load_path,
                                   dropout=0.1)

def build_trainer(model: Model,
                  train_loader: DataLoader,
                  dev_loader: DataLoader,
                  serialization_dir: str,
                  cuda_device: torch.device,
                  num_epochs: int,
                  patience: int,
                  num_gradient_accumulation_steps: int) -> Trainer:
    
    no_bigger = ["lstmencoder",
                 "gats",
                 "sym_dense",
                 "symptom_embedder",
                 "speaker_embedder",
                 "intends_embedder",
                 "actions_embedder"]
    
    parameter_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
            "weight_decay": 0.0,
            },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
            "lr": 0.0001
            }
        ]
    
    optimizer = AdamW(parameter_groups, lr=1e-5, eps=1e-8)
    lrschedule = LinearWithWarmup(optimizer=optimizer,
                                  num_epochs=num_epochs,
                                  num_steps_per_epoch=len(train_loader),
                                  warmup_steps=150)
    
    ckp = Checkpointer(serialization_dir=serialization_dir,
                       num_serialized_models_to_keep=-1)
    
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     patience=patience,
                                     validation_data_loader=dev_loader,
                                     validation_metric='+Micro-fscore',
                                     num_epochs=num_epochs,
                                     serialization_dir=serialization_dir,
                                     cuda_device=cuda_device if str(cuda_device) != 'cpu' else -1,
                                     learning_rate_scheduler=lrschedule,
                                     num_gradient_accumulation_steps=num_gradient_accumulation_steps,
                                     checkpointer=ckp)
    
    return trainer

def run_training_loop(config):
    
    serialization_dir = config.output_model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    os.makedirs(serialization_dir, exist_ok=True)
    
    dataset_reader = SymptomRecognitionDatasetReader(transformer_load_path=config.pretrained_model_dir,
                                                     symptom_norm_file=config.symptom_norm_file)
    train_path = config.train_file
    dev_path = config.dev_file
    train_data = list(dataset_reader.read(train_path))
    dev_data = list(dataset_reader.read(dev_path))
    vocab = build_vocab(train_data + dev_data)
    vocab.save_to_files(vocabulary_dir)
    
    train_loader = MultiProcessDataLoader(dataset_reader,
                                          train_path,
                                          batch_size=config.batch_size,
                                          shuffle=True)
    dev_loader = MultiProcessDataLoader(dataset_reader,
                                        dev_path,
                                        batch_size=config.batch_size,
                                        shuffle=False)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    
    device = torch.device(config.cuda_id if torch.cuda.is_available() else "cpu")
    model = build_model(vocab, config.pretrained_model_dir, len(dataset_reader.SymNorm2idx))
    model = model.to(device)
    
    trainer = build_trainer(model,
                            train_loader,
                            dev_loader,
                            serialization_dir,
                            device,
                            config.num_epochs,
                            config.patience,
                            config.num_gradient_accumulation_steps)
    
    trainer.train()
    return trainer

if __name__ == '__main__':
    init_logger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='./data/train_corpus.json', type=str)
    parser.add_argument("--dev_file", default='./data/dev_corpus.json', type=str)
    
    parser.add_argument("--symptom_norm_file", default='./data/dataset/symptom_norm.csv', type=str)
    parser.add_argument("--output_model_dir", default='./save_model', type=str)
    parser.add_argument("--pretrained_model_dir", default="./plms/roberta_base", type=str)
    
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--num_epochs", default=12, type=int)
    parser.add_argument("--patience", default=4, type=int)
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    
    config = parser.parse_args()
    run_training_loop(config)
    