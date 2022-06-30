# !pip install alive_progress
from scipy.io import savemat,loadmat
import scipy.io
import numpy as np
import os
from PIL import Image
import time
import sys
# from alive_progress import alive_bar
import bar

resolusi=256

folder="/content/drive/MyDrive/dataset_tes3"
folderrek="/content/drive/MyDrive/dataset_tes3_rekonstruksi_3125"
files=os.listdir(folder)
kelas=-1*np.ones((len(files),1))

toolbar_width = len(files)
# toolbar_width = 5
img_rz=np.ones((toolbar_width,resolusi,resolusi,3))


for i in bar.progressbar(range(toolbar_width), "Proses Membaca File: ", 85):
#for i in range(toolbar_width):
    j=i/(toolbar_width-1)*100
    if files[i][0]=='a':
        kelas[i,0]=0
    elif files[i][0]=='b':
        kelas[i,0]=1
    elif files[i][0]=='e':
        kelas[i,0]=2
    elif files[i][0]=='p':
        kelas[i,0]=3
#     print(os.path.join(folder,files[i]))
    image = Image.open(os.path.join(folder,files[i]))
    img_rz[i,:,:,:] = np.float64(np.array(image.resize((resolusi,resolusi))))
    sys.stdout.write(("\r [ %d"%j+"% ] "))
#     sys.stdout.write(('='*round(j))+(''*round((100-j)))+("\r [ %d"%j+"% ] "))
#   sys.stdout.flush()

# print(kelas)
# print(img_rz.shape)
print("Jumlah File :"+str(len(files)))
print("Resolusi :"+str(img_rz.shape[1])+"x"+str(img_rz.shape[2])+"x"+str(img_rz.shape[3]))
print("Jumlah Kelas a :"+str(np.sum(kelas==0)))
print("Jumlah Kelas b :"+str(np.sum(kelas==1)))
print("Jumlah Kelas e :"+str(np.sum(kelas==2)))
print("Jumlah Kelas p :"+str(np.sum(kelas==3)))
var1={'Citra':img_rz,'Kelas':kelas,'Resolusi':resolusi,'File':files,'FolderRek':folderrek}
savemat("datacitra.mat", var1)