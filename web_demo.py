import os
import sys
import torch
import gradio as gr
import torch.nn.functional as F
import torchvision.transforms as transforms
from train.build import device, model, path, IMG_SIZE

# 加载预训练模型
model_weights = os.path.join(path, "ckpt/resnet_90.18%.pth")
test_set = "test"

model.load_state_dict(torch.load(model_weights)["state_dict"])
model.eval()

# 垃圾类别
class_names = [
    "battery", "biological", "brown-grass", "carboard", "clothes", "green-grass",
    "metal", "paper", "plastic", "shoes", "trash", "white-grass"
]

# 图像预处理
preprocess = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


def classify_image(img):
    global model, preprocess

    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 预测图像
        outputs = model(img)
        confidences = F.softmax(outputs, dim=1)[0]
        confidences = {class_names[i]: float(confidences[i]) for i in range(12)}
    return confidences

# 构建 Gradio 界面
image_input = gr.Image(type="pil")
label_output = gr.Label(num_top_classes=12)

demo = gr.Interface(
    css=".footer {display:none !important}",
    title="垃圾分类智能识别系统",
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
)

demo.launch(server_name="127.0.0.1", server_port=50000, share=False)
