#filtro de gradiente 
import cv2 as cv2
import matplotlib.pyplot as plt
import json
from patchify import patchify
import numpy as np
import random
from scipy.ndimage import binary_fill_holes
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import time

################################################################################## 
###monta uma unica imagem 
def showSingleImage(img, title, size, mincolor=0, maxcolor=255):
    fig, axis = plt.subplots(figsize = size)

    axis.imshow(img, cmap='gray', vmin=mincolor, vmax=maxcolor)
    axis.set_title(title, fontdict = {'fontsize': 22, 'fontweight': 'medium'})
    plt.show()   
################################################################################## 
###retorna a imagem e os pontos dos circulos de marcação 
def load_sample(sample_fname, mode):

#ja conta com a imagem de tamanho redefinido
    if mode == 'train':

        img = cv2.imread(sample_fname + '.jpg')
        fg = cv2.imread(sample_fname + '.png')
        gt = fgtogt(fg)

        data = json.load(open(sample_fname.split('_resized')[0] + '.json'))
        pts = []
        circles = data['shapes']
        for circle in circles:
            pt = circle['points']
            pts.append([pt[0][0], pt[0][1], pt[1][0], pt[1][1]])

        return img, gt, pts
#redefine o tamanho da imagem para aplicar a mascara
    else:

        img = cv2.imread(sample_fname + '.jpg')
        nl, nc, nb = img.shape
        dsize = (int(0.2*nl), int(0.2*nc))
        img_resized = cv2.resize(img, dsize)
        
        data = json.load(open(sample_fname + '.json'))
        pts = []
        circles = data['shapes']
        for circle in circles:
            pt = circle['points']
            pts.append([pt[0][0], pt[0][1], pt[1][0], pt[1][1]])

        return img_resized, pts
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
###mostra o segundo maior numero de um array 
def  second_lnumber(list1):
    
    mx = max(list1[0], list1[1])
    secondmax = min(list1[0], list1[1])
    n = len(list1)
    
    for i in range(2,n):
            if list1[i] > mx:
                secondmax = mx
                mx = list1[i]
            elif list1[i] > secondmax and \
                mx != list1[i]:
                secondmax = list1[i]
            elif mx == secondmax and \
                secondmax != list1[i]:
                secondmax = list1[i]  
    return secondmax
##################################################################################
### salma igamem gerada
def save_resizeds(src_path):

    fnames = os.listdir(src_path)

    train_names = [fname.split('.')[0] for fname in fnames if fname.endswith('.png')]

    for i, train_name in enumerate(train_names):

        print(f'redimensionando e salvando a imagem {i+1} de um total de {len(train_names)} imagens...')

        train_fname = os.path.join(src_path, train_name)

        img = cv2.imread(train_fname + '.jpg')
        fg = cv2.imread(train_fname + '.png')

        nl, nc, nb = img.shape
        dsize = (int(0.2*nc), int(0.2*nl))
        img_resized = cv2.resize(img, dsize)
        fg_resized = cv2.resize(fg, dsize)

        cv2.imwrite(train_fname + '_resized.jpg', img_resized)
        cv2.imwrite(train_fname + '_resized.png', fg_resized)

    print('feito!')
##################################################################################   
def bool2mask(mask_tmp):
    mask = np.zeros(mask_tmp.shape, np.uint8)
    for l in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask_tmp[l,c] == True:
                mask[l,c] = 3 
            else:
                mask_tmp[l,c] = 2 
    return mask
##################################################################################
def fgtogt(fg):
    gt_b = fg[:,:,0]
    gt_b[gt_b > 0] = 1
    gt_g = fg[:,:,1]
    gt_g[gt_g > 0] = 1
    gt_r = fg[:,:,2]
    gt_r[gt_r > 0] = 1
    gt = gt_b * gt_g * gt_r
    return gt
##################################################################################    
def grabcut(img, maski):

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    maskf, bgdModel, fgdModel = cv2.grabCut(img, maski.astype('uint8') + 2, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    return maskf - 2
##################################################################################
def fillholes(maski):
    return binary_fill_holes(maski).astype('uint8')
##################################################################################
def save_segs(src_path, w):

    valid_names = [name for name in os.listdir(src_path) if 'resized' not in name]
    jpg_names = [name.split('.')[0] for name in valid_names if name.endswith('.jpg')]
    png_names = [name.split('.')[0] for name in valid_names if name.endswith('.png')]
    test_names = [name for name in jpg_names if name not in png_names]

    for i, test_name in enumerate(test_names):

        print(f'Segmentando a imagem {i} de um total de {len(test_names)} imagens...')

        img, pts = load_sample(os.path.join(src_path, test_name), mode='test')

        mask0 = segmenta(img, w, step=1)
        mask1 = grabcut(img, mask0)
        mask2 = fillholes(mask1)

        cv2.imwrite(os.path.join(src_path, test_name + '_segresized.png'), 255*mask2)

    print('Feito')    
##################################################################################   
def get_patches(img, gt, w, step):

    img_patches = patchify(img, (w, w, 3), step=step)
    gt_patches = patchify(gt, (w, w), step=step)

    img_patches0 = np.zeros((100, 3*w**2))
    img_patches1 = np.zeros((100, 3*w**2))

    idx0 = 0
    idx1 = 1

    npl = img_patches.shape[0]
    npc = img_patches.shape[1]

    random.seed(5489)
    sel_pls = random.sample(range(npl), npl)
    sel_pcs = random.sample(range(npc), npc)

    for l in sel_pls:
        for c in sel_pcs:
            if np.sum(gt_patches[l,c,:,:]) == 0 and idx0 < 100:
                img_patches0[idx0] = img_patches[l,c,0,:,:].reshape(-1)
                idx0 += 1
            elif np.sum(gt_patches[l,c,:,:]) == w**2 and idx1 < 100:
                img_patches1[idx1] = img_patches[l,c,0,:,:].reshape(-1)
                idx1 += 1
            if idx0 == 100 and idx1 == 100:
                break

    return img_patches0, img_patches1
##################################################################################
def save_patches(src_path, w, step):

    fnames = os.listdir(src_path)
    train_names = [fname.split('.')[0] for fname in fnames if fname.endswith('_resized.png')]

    for i, train_name in enumerate(train_names):
        print(f'extraindo os patches da imagem {i} de um total de {len(train_names)}...')
        img, gt, _ = load_sample(os.path.join(src_path, train_name),'train')
        patches0i, patches1i = get_patches(img, gt, w, step)
        if i == 0:
            patches0 = patches0i
            patches1 = patches1i
        else:
            patches0 = np.vstack((patches0, patches0i))
            patches1 = np.vstack((patches1, patches1i))

    print('feito!')

    np.save('patches0', patches0)
    np.save('patches1', patches1)
##################################################################################
def disp_img(img):

    plt.figure()
    if len(img.shape) == 3: # img is bgr
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else: # img is gray
        plt.imshow(img, cmap='gray')
    plt.show()
##################################################################################
def disp_sample(img, points):

    plt.figure()

    for point in points:
        center = tuple([int(point[0]), int(point[1])])
        r = int(((point[0] - point[2])**2 + (point[1] - point[3])**2)**(1/2))
        img = cv2.circle(img, center, r, [0, 0, 255], 3)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
##################################################################################
def segmenta(img, w, step):

    ti = time.time()

    img_patches = patchify(img, (w, w, 3), step=step)
    npl = img_patches.shape[0]
    npc = img_patches.shape[1]

    mask = np.zeros((img.shape[0], img.shape[1]))

    for l in range(npl):
        for c in range(npc):
            print(f'predizendo o rótulo do patch ({l},{c})...')
            mask[l,c] = clf.predict(img_patches[l, c, 0, :, :].reshape(1,-1))
    print('feito!')

    te = time.time() - ti

    print(f'Levou {te} segundos para segmentar esta imagem!')

    return mask
##################################################################################
###abre o dir e capta o nome das imgs.png

src_path =os.getcwd()
fnames = os.listdir(src_path)
train_names = [fname.split('.')[0] for fname in fnames if fname.endswith('.png')]
snames = train_names
real_list=[]
result_list=[]

##################################################################################
#craia mascara para aplicar no filtro grabcut
for i, train_name in enumerate(train_names):

    print(f'redimensionando e salvando a imagem {i+1} de um total de {len(train_names)} imagens...')

    train_fname = os.path.join(train_name)

    img = cv2.imread(train_fname + '.jpg')
    fg = cv2.imread(train_fname + '.png')

    nl, nc, nb = img.shape
    dsize = (int(0.2*nc), int(0.2*nl))
    img_resized = cv2.resize(img, dsize)
    fg_resized = cv2.resize(fg, dsize)

    cv2.imwrite(train_fname + '_resized.jpg', img_resized)
    cv2.imwrite(train_fname + '_resized.png', fg_resized)

print('feito!')

w = 5
step = 4*w

patches0 = np.load('patches0.npy')
patches1 = np.load('patches1.npy')
Xtrain = np.vstack((patches0, patches1))

lbls0 = np.zeros((len(patches0), 1))
lbls1 = np.ones((len(patches1), 1))
ytrain = np.vstack((lbls0, lbls1)).ravel()

clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))

print('treinando uma SVM...')
clf.fit(Xtrain, ytrain)
print('feito!')

save_segs(src_path, w)

  
##################################################################################
###le as imgs.png e aplica o filtro Canny

for sname in snames:
    numtabuas=0
    contagem_list=[]
  
    img, points = load_sample(os.path.join(sname),'1')
    seg = cv2.imread(os.path.join( sname + '.png'))
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    #showSingleImage(edges,sname,(12,7))
   # plt.show()
    

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
###faz um historiograma da coluna convoluida e elimina os valores iniciais e finais
##muito ruidosos
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
 

                  