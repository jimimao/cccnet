import torch
import torch.nn as nn
from utils import hparam
from torch.utils.data import DataLoader,Dataset,TensorDataset
import pandas as pd
import numpy as np
from model.model import CCCNet
import os
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 标记可用的显卡



class train_cccnet():
    def __init__(self,hp):
        self.hp = hp

    def data_load(self):
        hp = self.hp
        # 导入原始数据
        train_data = torch.from_numpy(np.array(pd.read_csv(hp.data.traindata_path))[:,0:-1])
        # print(train_data.size())  # torch.Size([48000, 36])
        train_label = torch.from_numpy(np.array(pd.read_csv(hp.data.traindata_path))[:,[-1]])
        # print(train_label.size())  # torch.Size([48000, 1])
        test_data = torch.from_numpy(np.array(pd.read_csv(hp.data.testdata_path))[:, 0:-1])
        # print(test_data.size())  # torch.Size([12000, 36])
        test_label = torch.from_numpy(np.array(pd.read_csv(hp.data.testdata_path))[:, [-1]])
        # print(test_label.size())  # torch.Size([12000, 1])

        # 转换 n* 36 -> n* 6*6
        train_data = train_data.view(-1,hp.data.data_dim,hp.data.data_dim)
        # print(train_data.size())  # torch.Size([48000, 6, 6])
        test_data = test_data.view(-1,hp.data.data_dim,hp.data.data_dim)
        # print(test_data.size())  # torch.Size([12000, 6, 6])

        train_dataset = TensorDataset(train_data.float(), train_label.float())
        test_dataset = TensorDataset(test_data.float(), test_label.float())

        train_loader = DataLoader(dataset=train_dataset, batch_size=hp.train.batch_size,
                                  shuffle=True, num_workers=hp.train.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=hp.test.batch_size,
                                 shuffle=False, num_workers=hp.test.num_workers)

        return train_loader, test_loader


    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hp = self.hp
        epoches = hp.train.epoches
        model = CCCNet(hp)  # load model CCCNet

        train_loader, test_loader = self.data_load()


        model.to(device)
        model.train()
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
        stepall = len(train_loader) # 每个epoch ，迭代完一次训练全集 的次数
        for epoch in range(hp.train.epoches):
            for i, dataset in enumerate(train_loader):
                data, label = dataset
                data = data.to(device)
                label = label.to(device)
                # print(i, data.size(),
                # label.size())
                # data : torch.Size([32, 6, 6])  [batch_size * dim * dim]
                # label : torch.Size([32, 1]) [batch_size * 1]
                out = model(data)
                predict = torch.max(out,1)[1] # 找出每行中值最大的数 ，并返回其列索引
                predict = predict.view(predict.size(0),-1).float()
                predict.requires_grad = True
                # print(label.size())
                loss = criterion(predict,label)
                if (i+1) % 100 == 0:
                    print("epoch: %d, step: %d / %d, loss: %.5f" % (epoch,i+1,stepall,loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()



        return 0






if __name__ == "__main__":
    # train
    hp = hparam.Hparam(file= './config/config.yaml')
    cccnet = train_cccnet(hp)
    cccnet.train()

    # save the model structure into tensorboard
    # input = torch.rand(32, 6, 6)
    # mm = CCCNet(hp)
    # with SummaryWriter(comment="CCCNet") as W:
    #     W.add_graph(mm,(input,))
    # # print(hp)