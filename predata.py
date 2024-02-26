import pandas as pd
from sklearn.model_selection import train_test_split

#get data train and test
data = pd.read_csv('/Users/jmac/Desktop/mip/CheXpert/test_labels.csv')
LABELS=data[['Path','Pleural Effusion']]
test_data=LABELS
#get data val
data1=pd.read_csv('/Users/jmac/Desktop/mip/CheXpert/val_labels.csv')
val_data=data1[['Path','Pleural Effusion']]
print(val_data)

#print(LABELS)
#train_data, test_data = train_test_split(LABELS, test_size=0.1, random_state=2019) #test size 10% overall
print(test_data)
#print(train_data)
