import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class CCCNet(nn.Module):  # v0.0.1
    def __init__(self,hp):
        super(CCCNet,self).__init__()
        self.hp = hp
        self.conv = nn.Sequential(
            ## upsample conv
            # H_out=(H_in−1)×stride[0]−2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
            # W_out=(W_in−1)×stride[0]−2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
            # in: (N,C_in,H_in,W_in)   (N,C_out,H_out,W_out)
            # [B,1,6,6]
            nn.ConvTranspose2d(1,64,kernel_size=(4,4), stride= 2, padding= 1),
            nn.BatchNorm2d(64),nn.ReLU(),
            #[B, 64, 12, 12]
            ## cnn1
            nn.ZeroPad2d((1,1,0,0)), #zeroPad2D (padding_left,padding_right,padding_top,padding_bottom)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,3),dilation=(1,1)),
            nn.BatchNorm2d(64),nn.ReLU(),
            ## cnn 2
            nn.ZeroPad2d((0,0,1,1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1),dilation=(1,1)),
            nn.BatchNorm2d(64),nn.ReLU(),
            # [B, 64, 12, 12]
            ## cnn 3
            nn.Conv2d(in_channels=64,out_channels=8,kernel_size=(1,1)),
            nn.BatchNorm2d(8),nn.ReLU(),
            # [B, 8, 12, 12]
            ## pooling
            nn.AvgPool2d(kernel_size=(2,2)),
            # [B, 8, 6, 6]
        )
        #lstm
        self.lstm = nn.LSTM(
            input_size= 6 * 6,
            hidden_size=hp.model.lstm_dim,
            batch_first=True,
            bidirectional= False,
        )
        # [B,hidden_size]

        # fc1
        self.fc1 = nn.Linear(8*hp.model.lstm_dim,hp.model.fc1_dim)
        # [B ,fc1_dim]
        #fc2
        self.fc2 = nn.Linear(hp.model.fc1_dim,hp.model.fc2_dim)
        # [B, fc2_dim]  # fc2_dim = 2
    def forward(self, x):
        # x : [B, data_dim, data_dim ]
        x = x.unsqueeze(1) # 第二维 增加一个维度
        # x : [B, channels ,data_dim, data_dim]
        x = self.conv(x)
        # x : [B, 8, data_dim, data_dim]
        x = x.view(x.size(0),x.size(1),-1)
        # x : [B, 8,data_dim*data_dim]
        x,_ =self.lstm(x)    # Outputs: output, (h_n, c_n)
        # x : [B,8,hidden_size]
        x = F.relu(x)
        x = x.contiguous().view(x.size(0), -1)
        # x : [B, 8*hidden_size]
        x = self.fc1(x)
        x = F.relu(x)
        # x : [B, fc1_dim]
        x = self.fc2(x)
        # x : [B, fc2_dim]
        # x = torch.sigmoid(x)
        return x

# class CCCNet(nn.Module):  #v 0.0.2
#     def __init__(self,hp):
#         super(CCCNet,self).__init__()
#         self.hp = hp
#         self.conv = nn.Sequential(
#             #[B, 64, 12, 12]
#             ## cnn1
#             nn.ZeroPad2d((1,1,0,0)), #zeroPad2D (padding_left,padding_right,padding_top,padding_bottom)
#             nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(1,3),dilation=(1,1)),
#             nn.BatchNorm2d(64),nn.ReLU(),
#             ## cnn 2
#             nn.ZeroPad2d((0,0,1,1)),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1),dilation=(1,1)),
#             nn.BatchNorm2d(64),nn.ReLU(),
#             # [B, 64, 12, 12]
#             ## cnn 3
#             nn.Conv2d(in_channels=64,out_channels=8,kernel_size=(3,3)),
#             nn.BatchNorm2d(8),nn.ReLU(),
#             # [B, 8, 12, 12]
#             ## pooling
#             # nn.AvgPool2d(kernel_size=(2,2)),
#             # [B, 8, 6, 6]
#         )
#
#         # fc1
#         self.fc1 = nn.Linear(8*16,64)
#         # [B ,fc1_dim]
#         #fc2
#         self.fc2 = nn.Linear(64,hp.model.fc2_dim)
#         # [B, fc2_dim]  # fc2_dim = 2
#     def forward(self, x):
#         # x : [B, data_dim, data_dim ]
#         x = x.unsqueeze(1) # 第二维 增加一个维度
#         # x : [B, channels ,data_dim, data_dim]
#         x = self.conv(x)
#         # x : [B, 8, data_dim, data_dim]
#         x = x.view(x.size(0),-1)
#         # x : [B, 8* 4]
#         x = self.fc1(x)
#         x = F.relu(x)
#         # x : [B, fc1_dim]
#         x = self.fc2(x)
#         # x : [B, fc2_dim]
#         x = torch.sigmoid(x)
#         return x
