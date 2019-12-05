import pandas as pd

df = pd.read_csv("./data/clinvar_conflicting_mapped.csv")

df1 = df[df['CLASS'] == 1]   #  16434 条
df1.index = [i for i in range(df1.shape[0])]
df0 = df[df['CLASS'] == 0]  #  48754 条
df0.index = [i for i in range(df0.shape[0])]

# 取60000条数据，20% 为test数据，80% 为train数据，比例依旧按照 1：3
# train数据{48000{1:12000,0:36000}}
# test 数据{12000{1:3000,0:9000}}
df_train = pd.concat([df1.loc[0:11999,:],df0.loc[0:35999,:]],axis=0)
# print(df_train.shape)
## 48000 * 37
df_test = pd.concat([df1.loc[12000:14999,:],df0.loc[36000:44999,:]],axis=0)
# print(df_test.shape)
## 12000 * 37

df_train.to_csv("./data/train_dataset.csv",index = 0)
df_test.to_csv("./data/test_dataset.csv",index = 0)