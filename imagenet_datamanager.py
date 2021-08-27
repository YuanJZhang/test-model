import os
import sys
import os.path as osp
from IPython import embed

from IPython.terminal.embed import embed

class ImageNet:
    dataset_dir = '01-imagenet'
    def __init__(self, root='data',**kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        self.test_dir = osp.join(self.dataset_dir,'testing-data')
        test_data = self.read_txt(self.test_dir, self.dataset_dir)
        self.testdata = test_data
    
    def read_txt(self, test_dir, data_dir):
        path = osp.join(data_dir, 'groundtruth.txt')
        gt_txt = open(path, 'r')
        lines = gt_txt.readlines()
        testdata = []
        
        for line in lines:
            path_label = line.strip().split()
            path_txt = path_label[0]
            label_txt = int(path_label[1].strip())
            img_path = osp.join(test_dir,path_txt)
            testdata.append((img_path, label_txt))
        print(len(testdata))
        return testdata

if __name__ == '__main__':
    imagenet = ImageNet(root='/home/zyj/test_model/data')