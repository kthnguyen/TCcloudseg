import cv2
import glob, os
import numpy as np
#%
#WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
WORKPLACE = r"K:\THEJUDGEMENT\BASINS_RESULTS_2012\SI"
filename = r"2012006S15043_CHANDA"
IRDIR = WORKPLACE + "\\"+ filename
os.chdir(IRDIR)
images = glob.glob("2012*.png")


frame = cv2.imread(IRDIR+ "\\" + images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter(filename+".avi", fourcc,15, (width,height),True)

i = 1
for image in images:
    video.write(cv2.imread(IRDIR+"\\"+image))
    print (image)

cv2.destroyAllWindows()
video.release()
print ('done')

#%%
WORKPLACE = r"K:"
IRDIR_1 = WORKPLACE +"\\"+ r"September2012"
IRDIR_2 = WORKPLACE +"\\"+ r"September2017"
os.chdir(IRDIR_1)
images_1 = glob.glob("*.png")
os.chdir(IRDIR_2)
images_2 = glob.glob("*.png")
os.chdir(IRDIR_1)

im1 = cv2.imread(IRDIR_1+ "\\" + images_1[0])
im2 = cv2.imread(IRDIR_2+ "\\" + images_2[0])
#im2 = cv2.resize(im2,(4200,1800))
im_concat = np.vstack((im2,im1))
height, width, layers = im_concat.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter("Compare2years.avi", fourcc,10, (width,height),True)

j = -1
for i in range(0,np.shape(images_2)[0]):
    im2 = cv2.imread(IRDIR_2+"\\"+images_2[i])
    im1 = cv2.imread(IRDIR_1+"\\"+images_1[i])
    im_concat = np.vstack((im2,im1))
    video.write(im_concat)
    print (images_2[i])

cv2.destroyAllWindows()
video.release()
print ('done')
#%% Combine 2 photos
WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR_1 = WORKPLACE + r"\CCMP_Wind\Remap\Figures"
IRDIR_2 = WORKPLACE + r"\GridSatB1_visible_IR\Remap_region\Combine_Figures"
os.chdir(IRDIR_1)
images_1 = glob.glob("*.png")
os.chdir(IRDIR_2)
images_2 = glob.glob("*.png")
os.chdir(IRDIR_1)

im1 = cv2.imread(IRDIR_1+ "\\" + images_1[0])
im2 = cv2.imread(IRDIR_2+ "\\" + images_2[0])
#im2 = cv2.resize(im2,(4200,1800))
im_concat = np.vstack((im2,im1))
height, width, layers = im_concat.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter("TCDorian_IR_VIS_WindVector.avi", fourcc,5, (width,height),True)

j = -1
for i in range(0,np.shape(images_2)[0]):
    im2 = cv2.imread(IRDIR_2+"\\"+images_2[i])
#    im2 = cv2.resize(im2,(4200,1800))
    if (i%12 == 0):
        j = j+1
    im1 = cv2.imread(IRDIR_1+"\\"+images_1[j])
    im_concat = np.vstack((im2,im1))
    video.write(im_concat)
    print (images_2[i])

cv2.destroyAllWindows()
video.release()
print ('done')