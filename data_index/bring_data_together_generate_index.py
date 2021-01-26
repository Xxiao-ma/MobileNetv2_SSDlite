import os 
import sys
from shutil import copyfile
data_path ='../../../data/Data/test/'

all_dirs = os.listdir(data_path)
all_dirs.sort()
train_dirs = all_dirs[:-6]
val_dirs = all_dirs[-6:]
test_dirs=all_dirs
print("train_dirs: "+str(train_dirs))
print("val_dirs: "+str(val_dirs))


if not os.path.exists('./test'):
    os.mkdir('./test')
    os.mkdir('./test/pic')
    os.mkdir('./test/label')

def generate_index(train_dirs, train_path,train):
    index= []
    f = open('./data/'+train+'.txt','w')
    for i in train_dirs:
        files = os.listdir(train_path+i)
        for j in files:
            if j[-7:]=='sub.pgm':
                index.append(j[:-8])
                copyfile(train_path+i+'/'+j, './test/pic/'+j)
                copyfile(train_path+i+'/'+j[:-8]+'.pgm', './test/pic/'+j[:-8]+'.pgm')
                if os.path.exists(train_path+i+'/'+j[:-8]+'.xml'):
                    copyfile(train_path+i+'/'+j[:-8]+'.xml', './test/label/'+j[:-8]+'.xml')
    print(len(index))
    for k in index:
        f.write(str(k)+'\n')
    f.close()

#generate_index(train_dirs, data_path, 'train')
#generate_index(val_dirs, data_path, 'val')
generate_index(test_dirs, data_path, 'test')



