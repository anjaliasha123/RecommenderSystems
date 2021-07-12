import pandas as pd
import numpy as np

data = pd.read_csv('a2.csv')

#print(data.columns)
data.index = data['document']
del data['document']
numattr = list(data['num-attr'])
del data['num-attr']
DF = data.tail(1)
data = data.drop(data.tail(1).index)
data['User 1'] = data['User 1'].fillna(0)
data['User 2'] = data['User 2'].fillna(0)

#print(data.index)
user1 = list(data['User 1'])
user2 = list(data['User 2'])
del data['User 1']
del data['User 2']
terms = list(data.columns)
documents = data.index

reqData = data[terms]
i = 0
for index,row in reqData.iterrows():
    reqData.loc[index] = reqData.loc[index]/(numattr[i]**0.5)
    i+=1
#print(reqData.head(2))
user1Terms = {}
user2Terms = {}
user1Score = {}
user2Score = {}

for term in terms:
    t = list(reqData[term])
    user1Terms[term] = np.dot(t,user1)
    user2Terms[term] = np.dot(t,user2)

#print("User 1 terms profile: \n {}".format(user1Terms))
#print("User 2 terms profile: \n {}".format(user2Terms))

#print(documents)
user1val = list(user1Terms.values())
user2val = list(user2Terms.values())
for doc in documents:
    user1Score[doc] = np.dot(user1val,reqData.loc[doc])
    user2Score[doc] = np.dot(user2val,reqData.loc[doc])


print("User 1 document profile: \n {}".format(user1Score))
print("User 2 document profile: \n {}".format(user2Score))

DF = DF.loc['DF'].tolist()[0:-2]
IDF = [ 1/x for x in DF]
print('IDF is:', IDF)

for doc in documents:
    user1Score[doc] = np.sum(user1val*reqData.loc[doc]*np.array(IDF))
    user2Score[doc] = np.sum(user2val*reqData.loc[doc]*np.array(IDF))


print("User 1 document profile with IDF: \n {}".format(user1Score))
print("User 2 document profile with IDF: \n {}".format(user2Score))