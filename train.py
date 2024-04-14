import torch

def train(train_dataloader, dev_dataloader, model, loss_fn, optimizer, epoches, device):
    for epoch in range(epoches):
        loss_list = []
        for idx, (input, label) in enumerate(train_dataloader):
            input = input.to(device)
            # print(input.size())
            pred_label = model(input)
            label = [int(item) for item in label]
            label = torch.tensor(label).to(device)
            loss = loss_fn(pred_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if (idx+1) % 1000 == 0:
                print(f"Epoch:{epoch+1}/{epoches}\tDataloader:{idx+1}的loss为:{loss.item()}")
        epoch_avg_loss = sum(loss_list)/len(loss_list)
        print(f"Epoch:{epoch+1}/{epoches}\t平均loss为:{epoch_avg_loss}")

        with torch.no_grad():
            correct = 0
            total = 0
            for input, label in dev_dataloader:
                input = input.to(device)
                label = [int(item) for item in label]
                total += len(label)
                pred_label = model(input)
                label = torch.tensor(label).to(device)
                _, predicted = torch.max(pred_label.data, 1)
                correct += (predicted == label).sum().item()
            print(f"Epoch:{epoch+1}/{epoches}\tdev数据:{total}, 预测正确数据:{correct}, 准确率为:{correct/(total)}")
        # save model
        torch.save(model.state_dict(), f"path of model save path/dataloader_model_{epoch+1}.pth")   