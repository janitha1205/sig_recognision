import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as img
import cv2

from matplotlib import pyplot as plt
import fir_filter
import pandas as pd
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz 
import scipy.signal

def ori(x,y):
  def rev(x,bias,up):
    temp=[]
    if up==1:
     
     for i in range(len(x)):
      temp.append(x[i]+bias)
     return temp
    else:
      for i in range(len(x)):
        temp.append(x[len(x)-i-1]+bias)
      return temp
  n= len(x)
  m=len(y) 
  temp=[]
  x1=rev(x,0,1)
  y1=rev(y,0,0)
  
  return x1,y1

    
    
def kalman_f(val):
  i=0
  pk=1
  pk1=1
  n=len(val)
  res=[]
  curvepre=np.zeros(n+1)
  for i in range(n):
    z1=val[i]
    f=1
    q=0.5
    pk=f*pk1*f+q
    r=0.1
    h=1
    s=r+h*pk*h
    kk=pk*h/s
    pk1=(1-kk*h)*pk
    curvepre[i+1]=f*curvepre[i]+kk*(z1-h*f*curvepre[i])
    res.append(curvepre[i+1])
  return res 


# read image C:\Users\user\Desktop\ecgpic.jpg
image = cv2.imread(r"ink.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 0, 180), (255, 255,255))
def inverte(imagem):
    imagem = abs(255-imagem)
    return imagem
cv2.imshow("Threshold Binary", inverte(mask))

cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()
# Step 2

def thin(im,dir):
   
     n= np.size(im,axis=0)
     m=np.size(im,axis=1)
 

     curx=[]
     cury=[]
     curx2=[]
     cury2=[]
     imres=np.zeros([n,m])
     if dir ==1:
       for j in range(n):
 #-1 0 1 -1 -1 -1
 #-1 0 1  0  0  0
 #-1 0 1  1  1  1
        last=0
        first=0
        hm=0
    
        for i in range(m):
      
         if (im[j,i]==255)&(hm==0):
          last=1
          first=i
          print(i)
         if (im[j,i]==255)&(hm==1):
          last=1
          first=i
          print(i)
      
         if  (im[j,i]<100)&(last==1)&(hm==0):
            imres[j,int((i+first)/2)]=255
            curx.append(j)
            cury.append(int((i+first)/2))
            last=0
            first=0
            hm=1
         else:
           if  (im[j,i]<100)&(last==1)&(hm==1):
            imres[j,int((i+first)/2)]=255
            curx2.append(j)
            cury2.append(int((i+first)/2))
            last=0
            first=0
            hm=0
     
     else :
          for j in range(m):
 #-1 0 1 -1 -1 -1
 #-1 0 1  0  0  0
 #-1 0 1  1  1  1
            last=0
            first=0
            hm=0
    
            for i in range(n):
      
              if (im[i,j]==255)&(hm==0):
                last=1
                first=i
              
                print(i)
              if (im[i,j]==255)&(hm==1):
                last=1
                first=i
           
                print(i)
      
              if  (im[i,j]<100)&(last==1)&(hm==0):
                imres[int((i+first)/2),j]=255
                cury.append(j)
                curx.append(int((i+first)/2))
                last=0
                first=0
                hm=1
              else:
                if  (im[i,j]<100)&(last==1)&(hm==1):
                   imres[int((i+first)/2),j]=255
                   cury2.append(j)
                   curx2.append(int((i+first)/2))
                   last=0
                   first=0
                   hm=0
       
     
     return imres,curx,cury,curx2,cury2
  


dilated = cv2.dilate(inverte(mask),np.ones([2,2]))
im=inverte(mask)

curve1,x,y,x1,y1 = thin(im,0)

# xi-------xf yi------yf normal
# xi------xf yf-----yi  inv(y)
# xf-------xi yi----yf inv(x)
# xf-------xi yf-----yi
def matchori(x,y,xn,yn):
  xi=[x[0],y[0]]
  xf=[x[len(x)-1],y[len(x)-1]]
  yi=[xn[0],yn[0]]
  yf=[xn[len(xn)-1],yn[len(xn)-1]]
  def dis(x,y):
    return abs(x[1]-y[1])+abs(x[0]-y[0])
  def inv(x):
    c=len(x)
    temp=[]
    for i in range(c):
      temp.append(x[c-i-1])
    return temp
  ind=np.argmin([dis(xf,yi),dis(yf,xf),dis(yi,xi),dis(xi,yf)])
  out=[]
  if ind==0:
    out=x
    out1=y
    out2=xn
    out3=yn
  if ind==1:
 
    out=x
    out1=y
    out2=inv(xn)
    out3=inv(yn)
  if ind==2:
    
    out=inv(x)
    out1=inv(y)
    out2=xn
    out3=yn
  if ind==3:

    out=inv(x)
    out1=inv(y)
    out2=inv(xn)
    out3=inv(yn)
  return out,out1,out2,out3
#curve2,x,y,x2,y2 = thin(im,1)


x3,y3,x32,y32=matchori(x,y,x1,y1)

print(x3)
print(y3)



cv2.imshow('0',curve1)

# For each contour, find the bounding rectangle and draw it






cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()


  

  


    
x,y=ori(x32,y32)
x1,y1=ori(x3,y3)

x.extend(x1)
y.extend(y1)
x_val=kalman_f(x)
y_val=kalman_f(y)

plt.plot(x_val,y_val)
plt.show() 