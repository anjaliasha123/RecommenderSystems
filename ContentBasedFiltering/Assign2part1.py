import pandas as pd
import numpy as np

data = pd.read_csv('a2.csv')

#print(data.columns)
data.index = data['document']
del data['document']
numattr = list(data['num-attr'])
del data['num-attr']
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

#print(user1)
#print(type(user2))
#print(terms)
user1Terms = {}
user2Terms = {}
user1Score = {}
user2Score = {}

for term in terms:
    t = list(data[term])
    user1Terms[term] = np.dot(t,user1)
    user2Terms[term] = np.dot(t,user2)

print("User 1 terms profile: \n {}".format(user1Terms))
print("User 2 terms profile: \n {}".format(user2Terms))

#print(documents)
user1val = list(user1Terms.values())
user2val = list(user2Terms.values())
for doc in documents:
    user1Score[doc] = np.dot(user1val,data.loc[doc])
    user2Score[doc] = np.dot(user2val,data.loc[doc])


print("User 1 document profile: \n {}".format(user1Score))
print("User 2 document profile: \n {}".format(user2Score))



