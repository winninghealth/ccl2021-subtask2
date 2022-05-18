## 上游任务标签

请将该数据集上游任务的标签预测结果保存在当前路径下

- 对话的疾病诊断标签：IMCS-DIAG_test.json
- 对话的意图识别标签：IMCS-IR_test.json（参考CBLUE榜单IMCS-IR任务的提交文件）
- 对话的实体识别标签：IMCS-NER_test.json（参考CBLUE榜单IMCS-NER任务的提交文件）

需要对实体识别标签进行症状归一化和格式转换后生成IMCS-NORM_test.json用于模型预测
