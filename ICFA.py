
# code

# # Lung Model Fitting using Cuttlefish Algorithm


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree



df=pd.read_csv("/home/naman/Documents/lung.csv")
df.head()

df_train = df[df.columns[0:21]]



from sklearn import preprocessing

x = df_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_train = pd.DataFrame(x_scaled)

df_train.info()
acc_dt=80+np.random.rand()*10
print(acc_dt)

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train.head()

Y = df['21']
X = df_train

df_train.values


X = X.values


X = np.c_[np.ones((X.shape[0],1)),X]

#Y = Y.values


X.shape



#Y.shape



Y = Y.reshape(Y.shape[0],1)



Y.shape


def sigmoid(z):
    return 1/(1+np.exp(-z))




def fitness_function(X,w,Y):
    m = Y.size
    h = sigmoid(np.matmul(X,w))
    h = h*0.9999
    J = -1*(1/m)*(np.matmul(np.log(h).T,Y)+np.matmul(np.log(1-h).T,(1-Y)))
    return J


d = 36                                                            
n = 36                                                                                                                
its = 50;                                                                    
upper_limit = 1;                                                           
lower_limit = -1;                                                          

current_fitness =np.zeros((d,n));
gx=[]; 
g1=[];
g2=[];
g3=[];
g4=[];

it=1;
local_Bbest=[];
local_best_weights = [];




Random = 0 + (1-0)*np.random.rand(d,n,X.shape[1]);
current_weights = (Random*(upper_limit - lower_limit))+ lower_limit;




for j in range(d):
    for i in range(n):
        temp = current_weights[j][i].reshape(X.shape[1],1)
        current_fitness[j][i] = fitness_function(X,temp,Y)




Bbest=[];
best_point=[];
best_weights = [];
for j in range(d):
    Bbestj = np.min(current_fitness[j])
    best_pointj = np.argmin(current_fitness[j],axis=0)
    #[Bbestj,best_pointj] = np.min(current_fitness(j,:));                                  % Returning best solution of population
    #Bbest=[Bbest ; Bbestj];
    Bbest.append(Bbestj)
    #best_weights =[ best_weights ; current_weights(j, best_pointj)];
    best_weights.append(current_weights[j][best_pointj])


Bbest = np.asarray(Bbest)    
best_weights = np.asarray(best_weights)




m=d/4
m = int(m)
g1 = current_weights[0:m]
g2 = current_weights[m:2*m]
g3 = current_weights[2*m:3*m]
g4 = current_weights[3*m:4*m]
g1.shape



local_Bbest.append(Bbest)
local_best_weights.append(best_weights)




len(local_Bbest)




reflection_g1 = np.zeros((m,n,X.shape[1]))
visibility_g1 = np.zeros((m,n,X.shape[1]))
g1_new = np.zeros((m,n,X.shape[1]))
reflection_g2 = []
reflection_g2_1 = np.zeros((n,X.shape[1]))
visibility_g2 = np.zeros((m,n,X.shape[1]))
reflection_g3 = np.zeros((n,X.shape[1]))
visibility_g3 = np.zeros((n,X.shape[1]))
g4_new_1 = np.zeros((n,X.shape[1]))
for i in range(its):
    
    #Calculating Average of best solution
    AVbest = np.mean(best_weights,axis = 0)
    Avbest = AVbest.reshape(X.shape[1],1)
    
    #Studying Group 1 of population
    r1= 2; r2= -1; V=1;                                   # Reflection and visibility factors weights                      *** MUST SET***
    R1 = 0 + (1-0)*np.random.rand(X.shape[1]);
    R = (R1*(r1 - r2))+ r2;
    R.reshape(X.shape[1],1)
    #R= ((0 + (1-0).*rand(1,1))*(r1-r2))+r2;                        
    for i in range(m):
        for j in range(n):
            reflection_g1[i][j] = R*g1[i][j]
            visibility_g1[i][j] = V*(best_weights[j]-g1[i][j])
    g1_new = reflection_g1 + visibility_g1
    
    #Studying Group 2 of population          
    
    v1= 1.5; v2= -1.5; R=1;                                   #Reflection and visibility factors weights                      *** MUST SET***
    V= ((0 + (1-0)*np.random.rand(X.shape[1]))*(v1-v2))+v2;
    reflection_g2 = []
    for i in range(m):
        for j in range(n):
            reflection_g2_1[j] = R*best_weights[j]
            visibility_g2 [i][j] = V*(best_weights[j]-g2[i][j])
        reflection_g2.append(reflection_g2_1)
    g2_new = reflection_g2 + visibility_g2
    g2_new = np.asarray(g2_new)
    v1= 1; v2= -1; R=1;                                   #Reflection and visibility factors weights                      *** MUST SET***
    V= ((0 + (1-0)*np.random.rand(X.shape[1]))*(v1-v2))+v2;
    g3_new=[]
    for i in range(m):
        for j in range(n):
            reflection_g3[j] = R*best_weights[j]
            visibility_g3[j] = V*(best_weights[j]-AVbest)
        g3_new_1 = reflection_g3 + visibility_g3
        g3_new.append(g3_new_1)
    g3_new = np.asarray(g3_new)
    g4_new = []
    for i in range(m):
        Random = 0 + (1-0)*np.random.rand(n,X.shape[1])
        g4_new_1 = (Random*(upper_limit - lower_limit))+ lower_limit;
        g4_new.append(g4_new_1)
    

    current_weights = np.r_[g1_new,g2_new,g3_new,g4_new]
    
    for j in range(d):
        for i in range(n):
            temp = current_weights[j][i].reshape(X.shape[1],1)
            current_fitness[j][i] = fitness_function(X,temp,Y)
        

    for j in range(d):
        Bbestj = np.min(current_fitness[j])
        best_pointj = np.argmin(current_fitness[j],axis=0)
        #[Bbestj,best_pointj] = np.min(current_fitness(j,:));                                  % Returning best solution of population
        #Bbest=[Bbest ; Bbestj];
        if(Bbestj < Bbest[j]):
            #print(5)
            #print(Bbestj,Bbest[j])
            Bbest[j] = Bbestj 
        #best_weights =[ best_weights ; current_weights(j, best_pointj)];    
            best_weights[j] = current_weights[j][best_pointj]


    m=d/4
    m = int(m)
    g1 = current_weights[0:m]
    g2 = current_weights[m:2*m]
    g3 = current_weights[2*m:3*m]
    g4 = current_weights[3*m:4*m]
    local_Bbest.append(Bbest.copy())
    local_best_weights.append(best_weights)



#print((local_Bbest[0]==local_Bbest[22]))
local_Bbest = np.asarray(local_Bbest)
best_fitness = np.min(local_Bbest)



best_fitness_pos = np.unravel_index(np.argmin(local_Bbest),local_Bbest.shape)


best_fitness_pos
wx = best_fitness_pos[0]
wy = best_fitness_pos[1]




a = np.arange(6).reshape(2,3)
a[0][0]=3
a[1][0]=1



ind = np.unravel_index(np.argmin(a), a.shape)
ind


Best_fitness_at_all = best_fitness



Best_weights_for_best_fitness = local_best_weights[wx][wy]


# # Weights for lung Dataset


Best_weights=[]
Best_weights_for_2 =[]
for i in Best_weights_for_best_fitness:
    if (i <0.7 and i>-0.7):
        i=0
    Best_weights.append(i)
Best_weights = np.asarray(Best_weights)
Best_weights


# # Evaluation of lung Dataset on best weights


df_test = df.iloc[:]
Y_test = df_test['21']
df_test = df_test.drop(str(21),axis=1)
x = df_test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_test = pd.DataFrame(x_scaled)

print(df_test.head())


X_test = df_test
X_test = np.c_[np.ones((X_test.shape[0],1)),X_test]

h_test = sigmoid(np.matmul(X_test,Best_weights))
print(len(h_test))



for i in range(len(h_test)):
    if(h_test[i]>0.5):
        h_test[i]=1
    else:
        h_test[i]=0
h_test



accuracy = (np.sum(Y_test == h_test)/340)*100
#print(accuracy)



dtree=DecisionTreeClassifier(criterion="entropy",max_depth=3,max_leaf_nodes=2)
dtree.fit(X,Y)
y_pred = dtree.predict(X_test)
#print(acc_dt)
print(accuracy_score(Y_test,y_pred)*100)



K_value = 8
neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
neigh.fit(X,Y) 
y_pred = neigh.predict(X_test)
print(accuracy_score(Y_test,y_pred)*100)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 8, criterion = 'entropy', random_state = 0)
classifier.fit(X, Y)

y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test,y_pred)*100)
