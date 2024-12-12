import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from captcha.image import ImageCaptcha
from random import randint,seed
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image

DIGIT=4


char_list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class CaptchaData(Dataset):
    def __init__(self,char_list,num):
        self.char_list=char_list
        self.char2index={
            self.char_list[i]:i for i in range (len(self.char_list))
        }
        self.label_list=[]
        self.img_list=[]
        self.type=type
        #Number of Cpatcha
        self.num=num

        for i in tqdm(range(self.num)):
            chars=""
            for i in range(DIGIT):
                chars+= char_list[randint(0,len(self.char_list)-1)]
            img =ImageCaptcha().generate_image(chars)

            self.img_list.append(img)
            self.label_list.append(chars)

    def __getitem__(self, index):
        chars=self.label_list[index]
        image=self.img_list[index].convert("L")
        chars_tensor=self._numerical(chars)
        img_tensor= self._totensor(image)
        #One Hot
        label=chars_tensor.long().unsqueeze(1)
        label_onehot=torch.zeros(DIGIT,36)
        label_onehot.scatter_(1,label,1)
        label=label_onehot.view(-1)

        return img_tensor,label

    def _numerical(self,chars):
        char_tensor=torch.zeros(DIGIT)
        for i in range(len(chars)):
            char_tensor[i]=self.char2index[chars[i]]
        return char_tensor
    def _totensor(self,img):
        return transforms.ToTensor()(img)

    def __len__(self):
        return self.num
    
train= CaptchaData(char_list,10000)
train_loader=DataLoader(train,batch_size=128,shuffle=True)
valid= CaptchaData(char_list,2000)
valid_loader=DataLoader(valid,batch_size=256,shuffle=False)

if __name__=="__main__":
    
    img,label=train[0]
    print(img)
    pred=torch.argmax(label.view(-1,36),dim=1)
    print(pred)
    plt.title([char_list[lab.int()] for lab in pred])
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

            