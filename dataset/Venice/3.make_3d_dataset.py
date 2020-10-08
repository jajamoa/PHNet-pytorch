import os,os.path,re,shutil,json
IMAGE_NUM = 2
SOURCE_PATH = '/.Venice/img_roi'
EXPORT_PATH = '/.Venice/ablation' + str(IMAGE_NUM)
ROI_PATH = '/.Venice/density_map_init'



def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)   
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.copyfile(srcfile,dstfile)      
        #print("copy %s -> %s"%( srcfile,dstfile))

def searchFile(pathname,filename):
    matchedFile =[]
    for root,dirs,files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append(file)
    return matchedFile

train_json = []
test_json = []

list1 = []
list1 = searchFile(SOURCE_PATH,'(.*).jpg')
list1.sort()

def normal(ind1, ind2):
    if ind1 < 0:
        return False
    if list1[ind1].split('_')[0] != list1[ind2].split('_')[0]:
        return False
    i1 = int(list1[ind1].split('.')[0].split('_')[1])
    i2 = int(list1[ind2].split('.')[0].split('_')[1])
    if (i2-i1) > 60*(ind2-ind1+1):
        print('{} lost more than one frame to {}'.format(list1[ind1], list1[ind2]))
        return False
    return True

for index in range(len(list1)):
    for place in range(IMAGE_NUM-1,-1,-1):
        if normal(index-place,index):
            mycopyfile(os.path.join(SOURCE_PATH, list1[index-place]), os.path.join(EXPORT_PATH, list1[index], list1[index-place]))
        else:
            mycopyfile(os.path.join(SOURCE_PATH, list1[index]), os.path.join(EXPORT_PATH, list1[index], list1[index].split('.')[0]+str(place)+'.jpg'))
    if list1[index].split('_')[0] == '4896':
        train_json.append(os.path.join(EXPORT_PATH, list1[index]))
    else:
        test_json.append(os.path.join(EXPORT_PATH, list1[index]))
            
        

with open('./jsons/train' + str(IMAGE_NUM) + '.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_json, json_file, indent=1)
    print(len(train_json))

with open('./jsons/test' + str(IMAGE_NUM) + '.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_json, json_file, indent=1)
    print(len(test_json))


