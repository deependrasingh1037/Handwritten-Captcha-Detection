import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

def get_segmented(img) : 
  kernel = np.ones((5,5))
  img3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 80)
  img3 = cv2.GaussianBlur(img3, (3, 3), 0)
  img3 = cv2.dilate(img3,kernel,iterations = 3)
  img3 = cv2.erode(img3,kernel,iterations = 1)
  gray_image = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)
  lst=[]
  for j in range(gray_image.shape[1]):
    sum = 0
    for i in range(gray_image.shape[0]):
      sum = sum+gray_image[i][j]
    lst.append(sum)
  x_cor = []
  f = 0
 
  for i in range(len(lst)):
    if lst[i] != 0 and f==0:
      x1=i
      f=1
    elif lst[i] == 0 and f==1:
      x2=i
      if(abs(x1-x2)>30):
        x_cor.append((min(x1,x2),max(x1,x2)))
      f=0 
  y_cor=[]
  for x1,x2 in x_cor:
    lst2=[]
    for i in range(gray_image.shape[0]):
      sum=0
      for j in range(min(x1,x2),max(x1,x2)+1):
        sum = sum+gray_image[i][j]
      lst2.append(sum)
    f=0
    for i in range(len(lst2)):
      if lst2[i] != 0 and f==0:
        y1=i
        f=1
      if lst2[i] == 0 and f==1:
        y2=i
        if(abs(y2-y1)>30):
          y_cor.append((min(y1,y2),max(y1,y2)))
        f=0 
  return img3, x_cor, y_cor

def Predict_Ans(image):
    model2 = load_model('team_dead_mosaic_ps1_folder.h5')
    class_mapping='ABEGHIKLMNSWXZ1234567'
    images, x_cor, y_cor=get_segmented(image)
    answer=""
    for i in range(len(x_cor)):
        image1=images[y_cor[i][0]:y_cor[i][1], x_cor[i][0]:x_cor[i][1]]
        image1=cv2.copyMakeBorder(image1,40,40,40,40,cv2.BORDER_CONSTANT)
        image1 = cv2.resize(image1, (28, 28))
        image1 = (np.array(image1)).reshape(1 , 28 , 28 , 1)
        result = np.argmax(model2.predict(image1))
        answer+=(class_mapping[result])
        
    return answer
if __name__=="__main__":
    print("Enter Image path")

    while True:
        path = input()
        if not os.path.exists(path):
            print("Enter Correct Path")
            continue
        img = cv2.imread(path)
        ans = Predict_Ans(img)
        print(ans)
        print("Enter next path for testing")
  
