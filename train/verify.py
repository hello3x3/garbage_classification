import os
import torch
from tqdm import tqdm
from build import device, model, dataloaders, path

model_weights = os.path.join(path, "ckpt/resnet_90.18%.pth")
test_set = "test"

model.load_state_dict(torch.load(model_weights)["state_dict"])
model.eval()

ac = 0

for inputs, labels in tqdm(dataloaders[test_set]):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        ac += (predictions == labels).sum()
        
rate = float(ac) / len(dataloaders[test_set].sampler)

print(f"正确率为：{rate*100:.2f}%")
