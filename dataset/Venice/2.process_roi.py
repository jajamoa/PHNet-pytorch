import os,os.path,re,shutil,h5py,cv2

SOURCE_PATH = '/.Venice/venice'
EXPORT_PATH = '/.Venice/img_roi'
ROI_PATH = '/.Venice/density_map_init'

IMAGE_NUM = 3

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)   
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.copyfile(srcfile,dstfile)     
        print("copy %s -> %s"%( srcfile,dstfile))

def searchFile(pathname,filename):
    matchedFile =[]
    for root,dirs,files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append((root,file))
    return matchedFile

train_json = []
test_json = []

list1 = []
list1 = searchFile(SOURCE_PATH,'(.*).jpg')

for index in range(len(list1)):
    roi_path = os.path.join(ROI_PATH, list1[index][1].replace('.jpg', '.h5'))
    data = h5py.File(roi_path, 'r')
    src = os.path.join(list1[index][0],list1[index][1])
    tar = os.path.join(EXPORT_PATH,list1[index][1])
    mycopyfile(src, tar)
    img = cv2.imread(tar)
    for i in range(3):
        img[:,:,i] = img[:,:,i] * data['roi']
    cv2.imwrite(tar,img)
    #break

