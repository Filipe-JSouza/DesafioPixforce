

#filtro de gradiente 
import cv2 as cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from scipy.stats import wilcoxon
import pandas as pd

################################################################################## 
###monta uma unica imagem 
def showSingleImage(img, title, size, mincolor=0, maxcolor=255):
    fig, axis = plt.subplots(figsize = size)

    axis.imshow(img, cmap='gray', vmin=mincolor, vmax=maxcolor)
    axis.set_title(title, fontdict = {'fontsize': 22, 'fontweight': 'medium'})
    plt.show()   
################################################################################## 
###retorna a imagem e os pontos dos circulos de marcação 
def load_sample(sample_fname):

    img = cv2.imread(sample_fname + '.jpg')
    
    data = json.load(open(sample_fname.split('_')[0] + '.json'))

    points = []
    circles = data['shapes']
    for circle in circles:
        point = circle['points']
        points.append([point[0][0], point[0][1], point[1][0], point[1][1]])

    return img, points
##################################################################################
###monta os circulos em suas respectivas imagens 
def display_sample(img, points):
    

    plt.figure()

    for point in points:
        center = tuple([int(point[0]), int(point[1])])
        r = int(((point[0] - point[2])**2 + (point[1] - point[3])**2)**(1/2))
        img = cv2.circle(img, center, r, [0, 0, 255], 3)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
##################################################################################   
### salma igamem gerada depois de aplicar grab cut
def grab_save(lista_imgs):
        
    
   rect=(400,400,2021,3000)   
  
   for sname in lista_imgs:
    img=cv2.imread( sname + '.jpg')
   
    mask= np.zeros(img.shape[:2], np.uint8)
    bgmodel= np.zeros((1,65),np.float64)
    fgmodel= np.zeros((1,65),np.float64)
    
    cv2.grabCut(img, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_grab = img*mask2[:,:,np.newaxis]
    cv2.imwrite(sname + '_grab.jpg', img_grab)
    
   snames = [fname.split('.')[0] for fname in fnames if fname.endswith('_grab.jpg')]
   return snames
 
##################################################################################
###abre o dir e capta o nome das imgs.png

src_path =os.getcwd()
fnames = os.listdir(src_path)
snames = [fname.split('.')[0] for fname in fnames if fname.endswith('.jpg')]
real_list=[]
result_list=[]

snames=grab_save(snames)

##################################################################################
###le as imgs.png e aplica o filtro Canny

for sname in snames:
    
    numtabuas=0
    contagem_list=[]
  
    img, points = load_sample(os.path.join(sname))
    seg = cv2.imread(os.path.join(sname)+'.jpg')
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    
    #showSingleImage(edges,sname,(12,7))
    #plt.show()
    

##################################################################################    
###faz conv para cada coluna da img afim de encontrar a distancia entre as linhas 
    ncolunas=len(edges[0])
    cols=[int(cols) for cols in np.linspace(48, ncolunas-1, 48)]
     
    xmaxx = 0
    xminn = 0
    
    for nlinhas in cols:
     
     
     x = edges[:, nlinhas].astype('float')
     x[x > 0] = 1

     #plt.plot(x,'ob')
     #plt.show()
     
     if sum(x)>0:
      xmin = np.where(x==1)[0].min()
      xmax = np.where(x==1)[0].max()
    
      y = abs(np.convolve(x, [-1, 1]))
      

      if sum(y)>0:

       z = np.where(y==1)[0]
       #plt.plot(z,'ob')
       #plt.show()
       w= abs(np.convolve(z, [-1, 1]))
       
##################################################################################
###Separa o ruido das bodas
       w[0]=0 
       w[len(w)-1]=0
      # plt.plot(w,'ob')
      # plt.show()
      
       tau=10
       u=[]
       
       for i,wi in enumerate(w):
            if i==0:
                u.append(wi)
            elif (wi-wlast)>=tau:
                u.append(wi)
            wlast=wi
            
       


##################################################################################
### calcula as espessura da tabua na linha
       espessura=np.median(u)
      
       
       if espessura >0:
            
        contagem=round((xmax-xmin)/espessura)
        contagem_list.append(contagem)
        numtabuas=round(np.median(contagem_list))
       else:
        contagem=0
        contagem_list.append(contagem)
        
        
    numtabuas=round(np.median(contagem_list))
      
 
    real_list.append(len(points))
    result_list.append(numtabuas)
     
    
     

##################################################################################
###wilcoxon para indentificar qualidade do codigo
d = {'imagens': snames, 'ideal': real_list, 'predito': result_list}
dado = pd.DataFrame(d)

ideal = np.array(dado['ideal'])

predito = np.array(dado['predito'])

wilcoxon(ideal, predito, zero_method='zsplit')
                  