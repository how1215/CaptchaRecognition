import torch 
from torch import nn,optim
from captcha_model import model
from captcha_data import train_loader,valid_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

epoch_lr=[
    (300,0.05),(100,0.001),(100,0.0001)
]
digit =4
device= torch.device("cuda")
criteron=nn.MultiLabelSoftMarginLoss()

def train():
    model.to(device)
    acces=[]
    losses=[]
    val_acces=[]
    val_losses=[]
    for n,(num_epoch,lr) in enumerate(epoch_lr):
        optimizer=optim.SGD(
            model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4
        )
        for epoch in range(num_epoch):
            model.train()
            epoch_loss=0.0
            epoch_acc=0.0
            for i,(img,label) in tqdm(enumerate(train_loader)):
                output=model(img.to(device))
                label=label.to(device)

                optimizer.zero_grad()

                loss=criteron(output,label)

                loss.backward()
                optimizer.step()

                pred=torch.argmax(output.view(-1,36),dim=1)
                true_lab=torch.argmax(label.view(-1,36),dim=1)

                epoch_acc+=torch.sum(pred==true_lab).item()
                epoch_loss+=loss.item()
            if epoch % 3 == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss=0.0
                    val_acc=0.0
                    for i,(img,label) in tqdm(enumerate(valid_loader)):
                        output=model(img.to(device))
                        label=label.to(device)
                        pred=torch.argmax(output.view(-1,36),dim=1)
                        true_lab=torch.argmax(label.view(-1,36),dim=1)
                        val_acc+=torch.sum(pred==true_lab).item()
                        val_loss+=loss.item()
                val_acc /= len(valid_loader.dataset)*digit
                val_loss/= len(valid_loader)
            epoch_acc /= len(train_loader.dataset)*digit
            epoch_loss/= len(train_loader)
            print(
                "epoch: {} , epoch loss {} , epoch accuracy {}".format(
                    epoch+sum([e[0] for e in epoch_lr[:n]]),epoch_loss,epoch_acc
                )

            )
            if epoch % 3 == 0:
                print(
                "epoch: {} , val loss {} , val accuracy {}".format(
                    epoch+sum([e[0] for e in epoch_lr[:n]]),val_loss,val_acc
                    )
                )
                for i in range(3):
                    val_acces.append(val_acc)
                    val_losses.append(val_loss)
            acces.append(epoch_acc)
            losses.append(epoch_loss)
            
    torch.save(model.state_dict(), "model_CNN.pth")

    print(losses)
    print(val_losses)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Train Loss')
    plt.plot(val_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acces, label='Train Accuracy')
    plt.plot(val_acces, label='Valid Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    train()