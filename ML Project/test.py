import numpy as np
import pandas as pd
from pandas import Series,DataFrame
client1 = ['Ahmed',2000,1200,'A','Ahmed@gamil.com',8.5]
client2 = ['Abdallah',2300,1300,None,'Abdallah@gamil.com',5.5]
client3 = ['hossam',5000,1000,'A',None,9]
client4 = ['Ali',7000,3000,'B','Mai@gamil.com',7.8]
client1 = ['Ahmed',2000,1200,'A','Ahmed@gamil.com',8.5]
client2 = ['Abdallah',2300,1300,None,'Abdallah@gamil.com',5.5]
client3 = ['hossam',5000,1000,'A',None,9]
client4 = ['Ali',7000,3000,'B','Mai@gamil.com',7.8]
df = pd.DataFrame([client1,client2,client3,client4], columns=['Name','Income','Withdraw','Class','Email','Score'])
print(df.loc[0:3,'Name':'Email'])
print("-----------------------------------------")
print(df.iloc[0:3,0:4])