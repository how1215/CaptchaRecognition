from captcha_model import model
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from captcha.image import ImageCaptcha
from random import randint,seed
from tqdm import tqdm
device= torch.device("cuda")
DIGIT=4
gt=[]
char_list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class CaptchaData(Dataset):
    def __init__(self,char_list,num):
        self.char_list=char_list
        self.char2index={
            self.char_list[i]:i for i in range (len(self.char_list))
        }
        self.label_list=[]
        self.img_list=[]
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

test= CaptchaData(char_list,num=100)
img,lab =test[0]


lab=lab.int().view(4,36).tolist()

column_indices = [j for row in lab for j, value in enumerate(row) if value == 1]

for c_index in column_indices:
    gt+=char_list[c_index]


model=model.to(device)
model.load_state_dict(torch.load("model_CNN.pth"))

model.eval()
prediction= model(img.unsqueeze(0).to(device)).view(4,36)
pred=torch.argmax(prediction,dim=1)

print("Predict :")
print([char_list[l.int()] for l in pred])
print("Ground Truth :")
print(gt)
plt.imshow(transforms.ToPILImage()(img))
plt.show()