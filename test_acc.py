import torch

import fruit_classify as p

path = r"G:\pythonProject\exercise\model\pvt_small.pth"

device = torch.device("cuda:0")
model = p.pvt_tiny().to(device)
model.load_state_dict(torch.load(path))
model.eval()

BATCH_SIZE = 8
total = 0
acc = 0

with torch.no_grad():  # 禁用梯度计算
    for batch_idx, (data, target) in enumerate(p.test_loader1, 0):
        data = data.to(p.device)
        target = target.to(p.device)
        outputs = model(data)

        predicted = (outputs.argmax(dim=1) == target).sum().item()
        acc += predicted
        total += BATCH_SIZE
        print("acc:{}".format(100*acc/total))


# accuracy = 100*acc / total
# print(f"准确率: {accuracy:.2f}%")








