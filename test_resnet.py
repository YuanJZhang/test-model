
import argparse
import time

from PIL import Image
from torch.autograd.grad_mode import F

from test_dataloader import ImageDataset
from imagenet_datamanager import ImageNet

import torch
import torch.cuda
from utils import Loggerring
import sys
import os
import os.path as osp
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision
from IPython import embed
from sklearn.metrics import accuracy_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='Train image model with center loss')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--root',type=str, default='/home/zyj/test_model/data/')
parser.add_argument('--save-dir', type=str, default='/home/zyj/test_model/log')
parser.add_argument('--batch_size', default=4, type=int,help="train batch size")
parser.add_argument('--workers', default=4, type=int,help="number of data loading workers (default: 4)")
args = parser.parse_args()

def main():
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.use_cpu: use_gpu = False
    sys.stdout = Loggerring(osp.join(args.save_dir, 'log_test.txt'))
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
    else:
        print("Currently using CPU (GPU is highly recommended)")
    dataset = ImageNet(root=args.root)
    test_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testloader = DataLoader(ImageDataset(dataset.testdata, test_transform),
    batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    val_num = len(dataset.testdata)
    model = torchvision.models.resnet50(pretrained=False)
    # model = torchvision.models.resnet101(pretrained=False)
    # model = torchvision.models.resnet152(pretrained=False)
    # model = torchvision.models.vgg16(pretrained=False)
    # model = torchvision.models.vgg19(pretrained=False)
    #print(model)

    #模型权重路劲
    model_weight_path = '/home/zyj/test_model/resnet50-pre.pth'
    assert osp.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()
    acc_top1 = 0.0
    acc_top5 = 0.0
    with torch.no_grad():
        for indx, (imgs, labels) in enumerate(testloader):
            imgs = imgs.to(device)
            labels = torch.unsqueeze(labels.to(device),dim=1)
            output = model(imgs)
            _, index1 = torch.topk(output,1)
            _, index5 = torch.topk(output,5)
            # a=torch.tensor([627, 627, 627, 627])
            # b=torch.tensor([654, 627, 785, 627])
            acc_top1 += (index1 == labels).sum().item()
            acc_top5 += (index5 == labels).sum().item()
    top1 = acc_top1/val_num
    top5 = acc_top5/val_num
        
    #计算平均推理时间
    model.eval()
    n = 0
    sum = 0.0
    y_ture = []
    y_pred = []
    init_imag = torch.zeros((1,3,224,224),device=device)
    #让模型先启动后面再做测试
    model(init_imag)
    avag_infer_time = 0.0
    with torch.no_grad():
        for (img_path, label) in dataset.testdata:
            img = Image.open(img_path).convert('RGB')
            img = test_transform(img)
            img = torch.unsqueeze(img, dim=0)
            start_time = time.time()
            output = model(img.to(device))
            infer_time = time.time()-start_time
            print("inferencen time: {}".format(infer_time))
            y_ture.append(label)
            y_pred.append(torch.topk(output,1).indices.item())
            sum = sum + infer_time
            n += 1
    avag_infer_time = sum / n

    print('top1:%.3f top5:%.3f' % (top1, top5))
    print("avaerage infertime: {}".format(avag_infer_time))
    print("accuracy:{}".format(accuracy_score(y_ture,y_pred)))
    print("percision:{}".format(precision_score(y_ture,y_pred,average='weighted')))
    print("recall:{}".format(recall_score(y_ture,y_pred,average='weighted')))
    


        
        


if __name__ == '__main__':
    main()