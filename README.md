## CCL2021 智能对话诊疗评测 症状识别

### 任务背景

任务名称：第一届智能对话诊疗评测比赛（第二十届中国计算语言学大会 CCL2021）

比赛结果：总分第1名（84.72%）

比赛任务：症状识别（任务二）第2名（76.214%）

比赛官网：http://www.fudan-disc.com/sharedtask/imcs21/index.html

方案思路：https://zhuanlan.zhihu.com/p/515695599

任务简介：针对互联网医患在线对话问诊的记录，该任务的目标是同时预测症状的归一化标签和类别标签。

### 数据集

IMCS21数据集由复旦大学大数据学院在复旦大学医学院专家的指导下构建。本次评测任务使用的IMCS-SR数据集在中文医疗信息处理挑战榜CBLUE持续开放下载，地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

CBLUE挑战榜公开的3,052条数据包括1,824条训练数据、616条验证数据和612条测试数据。请将下载后的数据（`IMCS_train.json`、`IMCS_dev.json`、`IMCS_test.json`和`symptom_norm.csv`）保存在文件夹`data/dataset`中。其中训练和验证数据来自CCL2021的训练集，测试数据来自CCL2021的测试集。

### 环境依赖

- 基于 Python (3.7.3+) & AllenNLP 实现

- 实验 GPU ：Tesla V100 & GeForce GTX 1080Ti

- Python 依赖：

```
torch==1.7.1+cu101
transformers==4.4.2
allennlp==2.4.0
```

### 快速开始

#### 预训练模型

使用下述2种不同规模的开源预训练模型：

1. chinese-roberta-wwm-ext，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext
2. chinese-roberta-wwm-ext-large，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

请将下载后的权重`pytorch_model.bin`保存在`plms`路径下相应名称的模型文件夹中

#### 数据预处理

```
python data_preprocess.py
```

- 对训练集和验证集数据进行预处理，生成`train_corpus.json`和`dev_corpus.json`，保存在`./data`中

#### 模型训练

```
python trainer.py --train_file ./data/train_corpus.json --dev_file ./data/dev_corpus.json --symptom_norm_file ./data/dataset/symptom_norm.csv --pretrained_model_dir ./plms/roberta_base --output_model_dir ./save_model --cuda_id cuda:0 --batch_size 2 --num_gradient_accumulation_steps 2 --num_epochs 12 --patience 4
```

- 参数：{train_file}: 训练集路径，{dev_file}: 验证集路径，{symptom_norm_file}: 症状归一化文件，{pretrained_model_dir}: 预训练模型路径，{output_model_dir}: 模型保存路径

#### 模型预测

```
python predict.py --test_input_file ./data/dataset/IMCS_test.json --test_output_file IMCS-SR_test.json --symptom_norm_file ./data/dataset/symptom_norm.csv --model_dir ./save_model --model_name best.th --pretrained_model_dir ./plms/roberta_base --cuda_id cuda:0 --dialogue_diagnosis ./data/imcs_results/IMCS-DIAG_test.json --dialogue_intention ./data/imcs_results/IMCS-IR_test.json --dialogue_symptom ./data/imcs_results/IMCS-NORM_test.json
```

- 参数：{test_input_file}: 测试集路径，{test_output_file}: 预测结果输出路径，{symptom_norm_file}: 症状归一化文件，{model_dir}: 加载已训练模型的路径，{model_name}：模型名称，{pretrained_model_dir}: 预训练语言模型的路径
- 上游任务的预测结果保存在`./data/imcs_results`路径下。{dialogue_diagnosis}为疾病标签的预测结果文件，{dialogue_intention}为意图标签的预测结果文件，在CBLUE榜单中为IMCS-IR任务的提交文件`IMCS-IR_test.json`，{dialogue_symptom}为症状实体识别的预测结果文件，需要通过对CBLUE榜单中IMCS-NER任务的提交文件`IMCS-NER_test.json`进行归一化和格式转换

#### 其他

- 症状归一化

  ```
  python entity_normalization.py
  ```

  对CBLUE榜单中的IMCS-NER任务预测结果进行症状归一化和格式转换。输入为`./data/imcs_results/IMCS-NER_test.json`，输出为`./data/imcs_results/IMCS-NORM_test.json`，作为模型预测的输入文件。注：归一化部分仅开源部分内容。

### 如何引用

```
@Misc{Jiang2022Shared,
      author={Yiwen Jiang},
      title={First Place Solutions of CCL2021, Symptom Recognition Task within Online Medical Dialogues},
      year={2022},
      howpublished={GitHub},
      url={https://github.com/winninghealth/ccl2021-subtask2},
}
```

### 版权

MIT License - 详见 [LICENSE](LICENSE)
