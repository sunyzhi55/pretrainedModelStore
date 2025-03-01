---
date: 2025-03-01T13:49:00
tags:
  - python
  - model
  - Clip 
---



# Clip Pretrained Model



文件夾`clip`記錄了各種網上已有的預訓練的`clip`模型

## 1、參考鏈接

1、`clip`的官方github鏈接：

[openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image](https://github.com/openai/CLIP)

2、`clip`模型大雜燴：

[mlfoundations/open_clip: An open source implementation of CLIP.](https://github.com/mlfoundations/open_clip)

3、`hugging Face`中關於`clip`的鏈接：

[huggingface.co](https://huggingface.co/docs/transformers/model_doc/clip)

## 2、代碼實現示例

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("device", device)
description = ["a diagram", "a dog", "a cat"]
model, preprocess = clip.load(r"./RN50x4.pt", device=device)

image = preprocess(Image.open(r"/path/of/img")).unsqueeze(0).to(device)
text = clip.tokenize(description).to(device)
# print("image", image.shape)  # image torch.Size([1, 3, 224, 224])
# print("text", text.shape)  # text torch.Size([3, 77])

with torch.no_grad():
    # 获取图像和文本的嵌入表示
    image_features = model.encode_image(image)
    # print("image_features", image_features.shape)  # image_features torch.Size([1, 512])
    text_features = model.encode_text(text)
    # print("text_features", text_features.shape)  # text_features torch.Size([3, 512])

    # 计算图像和文本的相似度
    logits_per_image, logits_per_text = model(image, text)
    # print("logits_per_image", logits_per_image, logits_per_image.shape)
    # print("logits_per_text", logits_per_text, logits_per_text.shape)
    """
    logits_per_image tensor([[19.8642, 25.7244, 19.0845]]) torch.Size([1, 3])
    logits_per_text tensor([[19.8642],
                            [25.7244],
                            [19.0845]]) torch.Size([3, 1])
    """
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)  # Label probs: [[0.00283868 0.9958597  0.00130165]]

    _, pred = torch.max(logits_per_image, dim=1)
    # print("pred", pred)  # pred tensor([1])
    print(f"This is {description[pred[0]]}")  # This is a dog
```

