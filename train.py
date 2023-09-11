import fruit_classify as p
import torch
epoch_num = 31
for epoch in range(epoch_num):
    p.train(p.model2,p.train_loader1, p.criterion, p.optimizer, epoch)
    if epoch % 5 == 0:
        # 保存模型
        path = "..\model\model_trans" + str(epoch) + ".pt"
        torch.save(p.model.state_dict(), path)