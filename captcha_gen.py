from captcha.image import ImageCaptcha
from random import randint,seed
import matplotlib.pyplot as plt


DIGIT=4

char_list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(len(char_list))

chars=''

for i in range(DIGIT):
    chars+= char_list[randint(0,35)]

img =ImageCaptcha().generate_image(chars)
print(img.size)
plt.imshow(img)
plt.show()