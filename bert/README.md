---
date: 2025-03-01T16:18:00
tags:
  - python
  - model
  - bert

---



# Bert Pretrained Model



文件夾`bert`記錄了各種網上已有的預訓練的`bert`模型

## 1、參考鏈接



1、[dmis-lab/biobert: Bioinformatics'2020: BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://github.com/dmis-lab/biobert)

## 2、代碼實現示例

```python
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
path = './BioClinical_BERT'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

# 输入文本
text = "age is 67, gender is male"

# 获取文本的嵌入表示
encoded_input = tokenizer(text, return_tensors='pt')
print("encoded_input", encoded_input)
outputs = model(**encoded_input)
# print("outputs", outputs)
# 模型的輸出
print("pooler_output", outputs.pooler_output.shape)
# 输出模型的隐藏状态
print("last_hidden_state", outputs.last_hidden_state.shape)

```

輸出結果：

```
encoded_input {'input_ids': tensor([[ 101, 1425, 1110, 5486,  117, 5772, 1110, 2581,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}
pooler_output torch.Size([1, 768])
last_hidden_state torch.Size([1, 9, 768])
```

> [!important]
>
> `BioClinical_BERT`裏面需要有`config.json`、`pytorch_model.bin`、`vocab.txt`

> 其中對應的`BioClinical_BERT`預訓練模型在對應的==release==中的
>
> `BioClinical_BERT_pretrained_model`目錄中