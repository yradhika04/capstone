#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd 
import sys
import csv
import time
import numpy as np


# In[22]:


'''Read csv file containing microgrid distance values '''

file = pd.read_csv("distance.csv") 
distance=pd.DataFrame(file).to_numpy()


# In[23]:


''' Next 3 cells for lottery DA '''


# In[24]:


'''Read csv file containing lottery DA winning buyers and sellers item values '''

file1 = pd.read_csv("lot.csv") 
lottery=pd.DataFrame(file1).to_numpy()
#print(lottery[0])
#res = list(eval(lottery[0][1]))
#print(res[1])


lottery_buyers=[]
lottery_sellers=[]
lottery_buyer_items=[]
lottery_seller_items=[]
templist=[]
tempbuyer=[]
tempseller=[]
nan=float('nan')
misc=[-1]


''' For buyers in lottery DA '''

for i in range (0, len(lottery)):
    templist=lottery[i][0]
    if(type(templist)==type(nan)):
        #print("check")
        lottery_buyers.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempbuyer.append(res[j][0])
    lottery_buyers.append(tempbuyer)
    tempbuyer=[]

tempbuyer=[]
print(len(lottery_buyers))


for i in range (0, len(lottery)):
    templist=lottery[i][0]
    if(type(templist)==type(nan)):
        #print("check")
        lottery_buyer_items.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempbuyer.append(res[j][1])
    lottery_buyer_items.append(tempbuyer)
    tempbuyer=[]

tempbuyer=[]
print(len(lottery_buyer_items))


''' For sellers in lottery DA '''

for i in range (0, len(lottery)):
    templist=lottery[i][1]
    if(type(templist)==type(nan)):
        #print("check")
        lottery_sellers.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempseller.append(res[j][0])
    lottery_sellers.append(tempseller)
    tempseller=[]

tempseller=[]
print(len(lottery_sellers))


for i in range (0, len(lottery)):
    templist=lottery[i][1]
    if(type(templist)==type(nan)):
        #print("check")
        lottery_seller_items.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempseller.append(res[j][1])
    lottery_seller_items.append(tempseller)
    tempseller=[]

tempseller=[]
print(len(lottery_seller_items))


# In[25]:


''' Calculate shortest distance mg for lottery DA results '''

mid=[]
lottery_answers=[]
least=sys.maxsize
temp=0
finali=0
finalj=0
finalk=0
misc=[-1]
start=0
end=0
lottery_time=[]

for i in range(0,len(lottery_buyers)):
    
    start=time.time()
    
    for j in range(0,len(lottery_buyers[i])):
        
        #start=time.time()

        for k in range(0,len(lottery_sellers[i])):
                        
            if(lottery_buyers[i][j]==lottery_sellers[i][k]):
                continue
            
            elif(lottery_buyers[i][j]==-1):
                lottery_answers.append(misc)
                end=time.time()
                lottery_time.append(end-start)
                continue
            
            else:
                
                x=lottery_buyers[i][j]-1
                y=lottery_sellers[i][k]-1
                
                if(x<y):
                    x=x+y
                    y=x-y
                    x=x-y

                if( (distance[x][y]<least) and (lottery_buyer_items[i][j]<= lottery_seller_items[i][k]) ):
                    least=distance[x][y]
                    temp=y
                    finali=i
                    finalj=j
                    finalk=k
    
        mid.append(temp)
        least=sys.maxsize
        
        lottery_seller_items[finali][finalk]=lottery_seller_items[finali][finalk]-lottery_buyer_items[finali][finalj]
            
    lottery_answers.append(mid)
    mid=[]
    
    end=time.time()
    lottery_time.append(end-start)
    
    
print(len(lottery_answers))
#print(lottery_answers)
#print(lottery_time)

tempmean=[]
for i in range(0, len(lottery_time)):
    tempmean.append(np.mean(lottery_time[i]))

mean=np.mean(tempmean)
print(mean)


# In[26]:


''' Writing lottery reslts and time in csv files '''

# writing final results
filename="lottery_answers.csv"
field1=[item for item in range(0,len(lottery_answers))]

with open(filename, 'w') as csvfile:
    #creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field1) #because panda dataframe reads 1st row as heading
    csvwriter.writerow(lottery_answers)
    
# reading time taken to calculate equilibirum point
file = pd.read_csv("lottery_time_normal.csv") 
data=pd.DataFrame(file).to_numpy()
 
# writing time taken to calculate equilibirum point + time taken to calculate least distance seller for each buyer   
filename="lottery_time_modified.csv"
field1=['Auction number', 'Time taken (s)']
row=[]
count=0

with open(filename, 'w') as csvfile:
    #creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field1) #because panda dataframe reads 1st row as heading
    
    for i in range(0, len(lottery_time)):
        row=[i,lottery_time[i]+data[i][1]]
        count=count+1
        csvwriter.writerow(row)
    
print(count)


# In[27]:


''' Next 3 cells for vickrey DA '''


# In[28]:


''' Read csv file containing vickrey DA winning buyers and sellers item values '''

file2 = pd.read_csv("vick.csv") 
vickrey=pd.DataFrame(file2).to_numpy()
#print(lottery[0][0])
#res = list(eval(lottery[0][1]))
#print(res[1][1])

vickrey_buyers=[]
vickrey_sellers=[]
vickrey_buyer_items=[]
vickrey_seller_items=[]
templist=[]
tempbuyer=[]
tempseller=[]
nan=float('nan')
misc=[-1]


''' For buyers in vickrey DA '''

for i in range (0, len(vickrey)):
    templist=vickrey[i][0]
    if(type(templist)==type(nan)):
        #print("check")
        vickrey_buyers.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempbuyer.append(res[j][0])
    vickrey_buyers.append(tempbuyer)
    tempbuyer=[]

tempbuyer=[]
print(len(vickrey_buyers))

for i in range (0, len(vickrey)):
    templist=vickrey[i][0]
    if(type(templist)==type(nan)):
        #print("check")
        vickrey_buyer_items.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempbuyer.append(res[j][1])
    vickrey_buyer_items.append(tempbuyer)
    tempbuyer=[]

tempbuyer=[]
print(len(vickrey_buyer_items))


''' For sellers in vickrey DA '''

for i in range (0, len(vickrey)):
    templist=vickrey[i][1]
    if(type(templist)==type(nan)):
        #print("check")
        vickrey_sellers.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempseller.append(res[j][0])
    vickrey_sellers.append(tempseller)
    tempseller=[]

tempseller=[]
print(len(vickrey_sellers))

for i in range (0, len(vickrey)):
    templist=vickrey[i][1]
    if(type(templist)==type(nan)):
        #print("check")
        vickrey_seller_items.append(misc)
        continue
    res = list(eval(templist))
    #print(len(res))
    for j in range(0,len(res)):
        tempseller.append(res[j][1])
    vickrey_seller_items.append(tempseller)
    tempseller=[]

tempseller=[]
print(len(vickrey_seller_items))


# In[29]:


''' Calculate shortest distance mg for vickrey DA results '''

mid=[]
vickrey_answers=[]
least=sys.maxsize
temp=0
finali=0
finalj=0
finalk=0
misc=[-1]
start=0
end=0
vickrey_time=[]


for i in range(0,len(vickrey_buyers)):
    
    start=time.time()
        
    for j in range(0,len(vickrey_buyers[i])):
        
        #start=time.time()
        
        for k in range(0,len(vickrey_sellers[i])):
            
            #sorted_items=vickrey_seller_items[k].sort()
            
            x=vickrey_buyers[i][j]-1
            y=vickrey_sellers[i][k]-1
            
            if(vickrey_buyers[i][j]==vickrey_sellers[i][k]):
                continue
            
            elif(vickrey_buyers[i][j]==-1):
                vickrey_answers.append(misc)
                end=time.time()
                vickrey_time.append(end-start)
                continue
            
            else:

                x=vickrey_buyers[i][j]-1
                y=vickrey_sellers[i][k]-1
                
                if(x<y):
                    x=x+y
                    y=x-y
                    x=x-y

                if((distance[x][y]<least) and (vickrey_buyer_items[i][j] <= vickrey_seller_items[i][k]) ):
                    least=distance[x][y]
                    temp=y
                    finali=i
                    finalj=j
                    finalk=k
                
        
        mid.append(temp)
        least=sys.maxsize
        
        vickrey_seller_items[finali][finalk]=vickrey_seller_items[finali][finalk]-vickrey_buyer_items[finali][finalj]
            
    end=time.time()
    vickrey_time.append(end-start)
    
    vickrey_answers.append(mid)
    mid=[]
    
print(len(vickrey_answers))
#print(vickrey_answers)
#print(vickrey_time)


tempmean=[]
for i in range(0, len(vickrey_time)):
    tempmean.append(np.mean(vickrey_time[i]))

mean=np.mean(tempmean)
print(mean)


# In[30]:


''' Writing vickrey reslts and time in csv files '''

# writing final results
filename="vickrey_answers.csv"
field1=[item for item in range(0,len(vickrey_answers))]

with open(filename, 'w') as csvfile:
    #creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field1) #because panda dataframe reads 1st row as heading
    csvwriter.writerow(vickrey_answers)

# reading time taken to calculate equilibirum point
file = pd.read_csv("vickrey_time_normal.csv") 
data=pd.DataFrame(file).to_numpy()
 
# writing time taken to calculate equilibirum point + time taken to calculate least distance seller for each buyer
filename="vickrey_time_modified.csv"
field1=['Auction number', 'Time taken (s)']
row=[]
count=0

with open(filename, 'w') as csvfile:
    #creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field1) #because panda dataframe reads 1st row as heading
    
    for i in range(0, len(vickrey_time)):
        row=[i,vickrey_time[i]+data[i][1]]
        count=count+1
        csvwriter.writerow(row)
    
#sum(vickrey_buyer_items[i]
print(count)


# In[31]:


filename="basic.csv"
field1=['Auction number', 'Buyer energy unit requirements']
row=[]
count=0

with open(filename, 'w') as csvfile:
    #creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(field1) #because panda dataframe reads 1st row as heading
    
    for i in range(0, len(vickrey_answers)):
        row=[i,sum(vickrey_buyer_items[i])]
        count=count+1
        csvwriter.writerow(row)


# In[ ]:




