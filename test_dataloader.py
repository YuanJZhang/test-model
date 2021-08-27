from PIL import Image
import os.path as osp

from torch.utils.data.dataset import Dataset


def read_image(imgpath):
    got_img = False

    if not osp.exists(imgpath):
        raise IOError("{}not exits".format(imgpath))
    while not got_img:
        try:
            img = Image.open(imgpath).convert('RGB')
            got_img = True
        except IOError:
            print("{} not read image".format(imgpath))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(ImageDataset,self).__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label