from PIL import Image
from numpy import asarray
import numpy as np
from matplotlib import pyplot
from scipy.fftpack import dct 
import math as mt
from scipy.fft import dctn, idctn
from scipy.io import savemat,loadmat
import recons
import bar
import time,sys
import os

resolusi=256
persen=0.03125
windowsize_r = 8
windowsize_c = 8
thr=20

# folderrek="E:\TA\dataset_rekonstruksi_sedang"
# folderrek="E:\TA\dataset_rekonstruksi_sedikit"
# folderrek='E:\TA\dataset_rekonstruksi_sedang'

datacitra=loadmat("datacitra.mat")
img_rz=datacitra['Citra']
kelas=datacitra['Kelas']
resolusi=datacitra['Resolusi']
files=datacitra['File']
folderrek=datacitra['FolderRek']

# print(folderrek.dtype)
# print(files)
# print(folderrek)
# x=os.path.join(folderrek,files[1])
# print(x)

L=windowsize_r*windowsize_c
M=round(persen*windowsize_r*windowsize_c)
y=np.ones((M,round(img_rz.shape[1]*img_rz.shape[2]*img_rz.shape[3]/L),img_rz.shape[0]))
ysave=np.ones((img_rz.shape[0],M,round(img_rz.shape[1]*img_rz.shape[2]*img_rz.shape[3]/L)))

img_rr=np.ones((img_rz.shape[0],img_rz.shape[1],img_rz.shape[2],img_rz.shape[3]))
snr=np.ones((1,img_rz.shape[1]*img_rz.shape[2]*img_rz.shape[3]))

folderrek="/content/drive/MyDrive/dataset_tes3_rekonstruksi_3125"
# print(str(img_rz.shape[3]))
# for j in range(0,img_rz.shape[3],1):
for j in bar.progressbar(range(img_rz.shape[0]), "Proses CS: ", 85):
#for j in range(img_rz.shape[0]):
    jj=j/(img_rz.shape[0]-1)*100
#     print(str(j))
#   ind=0
    sfile=False
    while sfile==False:
        single=False
        np.random.seed()
        AA=np.random.normal(0,1,(M,L))
        # print("A1="+str(AA))
        md={'A':AA}
        savemat("dataA.mat", md)
        ind=0
        for i in range(0,img_rz.shape[3],1):
#             print(str(i))
            for r in range(0,img_rz.shape[1] , windowsize_r):
                for c in range(0,img_rz.shape[2] , windowsize_c):
                    w = img_rz[j,r:r+windowsize_r,c:c+windowsize_c,i]
#                     wid1=w.reshape(windowsize_r*windowsize_c,1)
                    wd2=dctn(w[:,:])
                    wd=wd2.reshape(windowsize_r*windowsize_c,1)
                    data_dict=loadmat("dataA.mat")
                    AA=data_dict['A']
                    y[0:M,ind:ind+1,j]=np.dot(AA,wd)
                    Y=y[0:M,ind:ind+1,j]
                    ysave[j,0:M,ind:ind+1]=y[0:M,ind:ind+1,j]
                    try:
                        hat_x1=recons.omp(Y,AA)
                        
                    except:
                        print('Singular')
                        single=True
                        break
                    snr[0,ind]=np.mean(np.mean(wd**2))/np.mean(np.mean((wd-hat_x1.T)**2))
#                     hat_xt=hat_x1.reshape(windowsize_r*windowsize_c,1)

                    ind=ind+1
                    wdir=(hat_x1.T).reshape(windowsize_r,windowsize_c)
                    wid=idctn(wdir)
                    img_rr[j,r:r+windowsize_r,c:c+windowsize_c,i]=wid
                    
                if single==True:
                    break
 #               if np.sum(snr>30)<thr:
 #                   break
            if single==True:
                break
        mse=np.mean(np.mean((img_rz[j,:,:,:]-img_rr[j,:,:,:])**2))
        psnr=10*mt.log10(255**2/mse)

        if psnr>10:
          sfile=True

#          if np.sum(snr>30)<thr:
#                break
#           else:
#                sfile=True
#                 print('True')
#     print(files[j])
    
    sys.stdout.write(("\r [ %d"%jj+"% ] "))
#    sys.stdout.flush()   
    I0 = img_rz[j,:,:,:].astype(np.uint8)
    I = img_rr[j,:,:,:].astype(np.uint8)
    Is = Image.fromarray(I)
   
#     im1 = Is.save(os.path.join(folderrek,files[j]))
#     print(files[j].strip)
    im1 = Is.save(os.path.join(folderrek,files[j].strip()))
    mse=np.mean(np.mean((img_rz[j,:,:,:]-img_rr[j,:,:,:])**2))
#   mse=np.mean(np.mean((I-I0)**2))
    psnr=10*mt.log10(255**2/mse)
#    print("PSNR = "+str(psnr)+" dB")

print("Rasio Kompresi ="+str(persen*100)+"%")

fig, axs = pyplot.subplots(1,2)

axs[0].imshow(I0)
axs[0].set_title('Citra Asli')
axs[1].imshow(I)
axs[1].set_title('Citra Rek')

# axs[0].plot(wd)
# axs[0].set_title('Citra Asli')
# axs[1].plot(hat_xt)
# axs[1].set_title('Citra Rek')