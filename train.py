import torch
import torch.nn as nn
from utils import hparam
from torch.utils.data import DataLoader,Dataset,TensorDataset
import pandas as pd
import numpy as np
from model.model import CCCNet
import os
import logging
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
        writer = SummaryWriter(comment = 'CCCNet')
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     handlers=[
        #         logging.FileHandler(os.path.join(log_dir,
        #                                          '%s-%d.log' % (args.model, time.time()))),
        #         logging.StreamHandler()
        #     ]
        # )
        # logger = logging.getLogger()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hp = self.hp
        epoches = hp.train.epoches
        model = CCCNet(hp)  # load model CCCNet

        train_loader, test_loader = self.data_load()


        model.to(device)
        # model.train()
        criterion = nn.MSELoss().to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
        stepall = len(train_loader) # 每个epoch ，迭代完一次训练全集 的次数
        for epoch in range(hp.train.epoches):
            model.train()
            total_loss = 0
            total = 0
            for i, dataset in enumerate(train_loader):
                data, label = dataset[0].to(device),dataset[1].to(device)
                # print(i, data.size(),
                # label.size())
                # data : torch.Size([32, 6, 6])  [batch_size * dim * dim]
                # label : torch.Size([32, 1]) [batch_size * 1]
                out = model(data)
                predict = torch.max(out,1)[1] # 找出每行中值最大的数 ，并返回其列索引
                predict = predict.view(predict.size(0),-1).float()
                predict.requires_grad = True
                # print(label.size())
                # label = label.long()
                loss = criterion(predict,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.size(0)
                total += data.size(0)
                if (i+1) % 100 == 0:
                    print("epoch: %d, step: %d / %d, total_loss: %.5f" % (epoch,i+1,stepall,total_loss/total))
                    writer.add_scalar('Train Loss', total_loss/total, epoch*stepall+i+1)
            if (epoch+1) % hp.train.chkpt == 0:
                if not os.path.exists(hp.log.chkpt_dir):
                    os.mkdir(hp.log.chkpt_dir)
                save_path = os.path.join(hp.log.chkpt_dir,'chkpt_%d.pt'%(epoch))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, save_path)
                # logger.info('Save model checkpoint at epoch: %d' % epoch)
                # logger.info('-'*20)
                print('Save model checkpoint at epoch: %d' % epoch)
                print('-' * 20)

                # load model
                # checkpoint = torch.load(PATH)
                # model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # epoch = checkpoint['epoch']
                # loss = checkpoint['loss']
                #
                # model.eval()
                # # - or -
                # model.train()
                # logger.info('Validate the test dataset')
                print('Validate the test dataset')
                model.eval()
                rightnum = 0
                numsum = 0
                for j,y_dataset in enumerate(test_loader):
                    y_data,y_label = y_dataset[0].to(device),y_dataset[1].to(device)
                    y_out = model(y_data)
                    y_predict = torch.max(y_out,1)[1]
                    y_predict = y_predict.view(y_predict.size(0) ,-1)
                    rightnum += y_label.eq(y_predict.data).cpu().sum().item()
                    numsum += y_data.size(0)
                writer.add_scalar('Test',rightnum / numsum ,epoch)
                print("test accuracy: %.5f" % (rightnum/ numsum))
                # logger.info("test accuracy: %.5f" % rightnum/numsum)


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