from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from numba import njit

#%%
#function for dependency between m th and n th sensor

def dep_cor(rsq_mat,cumsum,m,n):
  import statistics
  ind1=list(range(cumsum[m-1],cumsum[m]))
  ind2=list(range(cumsum[n-1],cumsum[n]))
  cor=rsq_mat.iloc[ind1,ind2]
  return(statistics.mean(cor.apply(max,1)))


#%%
def func_modified(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals, yvals, reduction):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split
  import statistics
  import matplotlib.pyplot as plt
  
  xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,random_state=None, test_size=0.2,  shuffle=True)
  print("Dataset splitted")                                                                   
  starter_learning_rate = 0.001
  num_features=sum(sensor_sizes)
  nrow=len(yvals_train)
  num_output=num_classes

  input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
  input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

  s=tf.compat.v1.InteractiveSession()
  ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
  weights_0 = tf.Variable(tf.random.normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
  bias_0 = tf.Variable(tf.random.normal([num_layers_0]))
  weights_1 = tf.Variable(tf.random.normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
  bias_1 = tf.Variable(tf.random.normal([num_output]))

  ## Initializing weigths and biases
  hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
  predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

  ##calculate penalty terms
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  cumsum =[0]+ list(series.cumsum())
  ##calculate penalty terms
  if(regularizer_rate_1!=0):
      penalty=(tf.reduce_sum(tf.square(weights_0[0:sensor_sizes[0]])))**0.5/sensor_sizes[0]
      for i in range(len(sensor_sizes)-1):
        penalty=penalty+((tf.reduce_sum(tf.square(weights_0[cumsum[i+1]:cumsum[i+2]])))**0.5)/sensor_sizes[i+1]
  else:
       penalty=0
  print("GL value calculated")    
  dep_mat=[[0 for i in range(len(sensor_sizes))] for j in range(len(sensor_sizes))]
  redund=0
  if(regularizer_rate_0!=0):   
      r_mat=np.array(xvals_train.corr())
      rsq_mat=[[elem*elem for elem in inner] for inner in r_mat]
      rsq_mat=pd.DataFrame(rsq_mat)
      for i in range(len(sensor_sizes)):
        for j in range(len(sensor_sizes)):
          if j!=i:
            dep_mat[i][j]=dep(rsq_mat,cumsum,i+1,j+1)
            #redund=redund+dep(rsq_mat,cumsum,i+1,j+1)*((tf.reduce_sum(tf.square(weights_0[cumsum[j]:cumsum[j+1]])))*(tf.reduce_sum(tf.square(weights_0[cumsum[i]:cumsum[i+1]])))**0.5)/(sensor_sizes[i]*sensor_sizes[j])
            redund=redund+dep_mat[i][j]*(tf.reduce_sum(tf.square(weights_0[cumsum[i]:cumsum[i+1]])))**0.5/sensor_sizes[i]
    
      if len(sensor_sizes)>1:
        redund=redund/(len(sensor_sizes)*(len(sensor_sizes)-1))
  print("dep value calculated")      
  mse=tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) 
  loss = mse + regularizer_rate_0*redund/num_layers_0 + regularizer_rate_1*penalty/num_layers_0 
  print("loss calculated")

  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  ## Metrics definition
  correct_prediction = tf.equal(tf.argmax(yvals_train,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  training_accuracy = []
  training_loss = []
  r=[]
  p=[]
  ms=[]

  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  weight=[]
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:xvals_train, 
                                                         input_y: yvals_train}))
    training_loss.append(s.run(loss, {input_X: xvals_train, 
                                      input_y: yvals_train}))
    ms.append(s.run(mse, {input_X: xvals_train, 
                                      input_y: yvals_train}))
    if(regularizer_rate_0!=0):
        r.append(s.run(redund, {input_X: xvals_train, 
                                      input_y: yvals_train}))
    if(regularizer_rate_1!=0):
        p.append(s.run(penalty, {input_X: xvals_train, 
                                      input_y: yvals_train}))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,
                                                                   training_loss[epoch],
                                                                    training_accuracy[epoch]
                                                                   ))
    w0=weights_0.eval()
    w=[]
    #w.append(norm(w0[0:sensor_sizes[0]],2))
    for i in range(len(sensor_sizes)):
      w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
    weight.append(w)
     
  r=[regularizer_rate_0*i/num_layers_0**2 for i in r]  
  p=[regularizer_rate_1*i/num_layers_0 for i in p]   
  
  
  y_pred = np.array(s.run(predicted_y, feed_dict={input_X: xvals_test}))
  y_pred= np.where(y_pred == y_pred.max(axis=1)[:, np.newaxis], 1, 0)
  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))

  w0=weights_0.eval()
  w=[]
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2)/sensor_sizes[i])
  print(w)
  if reduction==True:
      weight=pd.DataFrame(weight)
      plt.plot(weight[0], label = "Feature 1")
      plt.plot(weight[1], label = "Feature 2")
      plt.plot(weight[2], label = "Feature 3")
      plt.plot(weight[3], label = "Feature 4")
      plt.xlabel("Iteration")
      plt.ylabel("Norm of the input layer weights")
      plt.legend()
      #plt.savefig("weights-iris.jpeg")
      plt.show()
    
  #Feature selection
  if reduction==True:
    v=[i for i,x in enumerate(w) if x > 0.1*max(w)]
    selected=[]
    for i in v:
      selected.append(xvals.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_reduced=pd.concat(selected,ignore_index=True, axis=1)
    print("reduced data formed")
    acc=[]
    sensor_sizes_red=[sensor_sizes[i] for i in v]
    s.close()
    for i in range(10):
     xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals_reduced,yvals,random_state=None, test_size=0.2,  shuffle=True)
     x=func_modified_landsat(0,0,num_layers_0, epochs, batch_size, num_classes, sensor_sizes_red,dep_cor,xvals_train, yvals_train,xvals_test, yvals_test,reduction=False)
     acc.append(x[0])
    
    return([sum(acc)/10,statistics.stdev(acc),len(sensor_sizes_red),v,dep_mat])
  else:
    s.close()
    return([testacc,len(sensor_sizes),dep_mat])



#%%
def func_modified_landsat(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals_train, yvals_train,xvals_test, yvals_test, reduction):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split

  #xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,random_state=None, test_size=0.2,  shuffle=True)
                                                                     
  starter_learning_rate = 0.001
  num_features=sum(sensor_sizes)
  nrow=len(yvals_train)
  num_output=num_classes

  input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
  input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

  s=tf.compat.v1.InteractiveSession()
  ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
  weights_0 = tf.Variable(tf.random.normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
  bias_0 = tf.Variable(tf.random.normal([num_layers_0]))
  weights_1 = tf.Variable(tf.random.normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
  bias_1 = tf.Variable(tf.random.normal([num_output]))

  ## Initializing weigths and biases
  hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
  predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

  ##calculate penalty terms
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  
  cumsum =[0]+ list(series.cumsum())
  
  
  loss = tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) 

  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  ## Metrics definition
  correct_prediction = tf.equal(tf.argmax(yvals_train,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  training_accuracy = []
  training_loss = []

  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:xvals_train, 
                                                         input_y: yvals_train}))
    training_loss.append(s.run(loss, {input_X: xvals_train, 
                                      input_y: yvals_train}))
  print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch]
                                                                   ))
    
  
    
  y_pred = np.rint(s.run(predicted_y, feed_dict={input_X: xvals_test}))

  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))

  w0=weights_0.eval()
  w=[]
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
  print(w)
  #Feature selection
  if reduction==True:
    v=[i for i,x in enumerate(w) if x > 0.1*max(w)]
    selected=[]
    for i in v:
      selected.append(xvals_train.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_train_red=pd.concat(selected,ignore_index=True, axis=1)
    
    selected=[]
    for i in v:
      selected.append(xvals_test.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_test_red=pd.concat(selected,ignore_index=True, axis=1)
    
  
    acc=0
    c=0
    sensor_sizes_red=[sensor_sizes[i] for i in v]
    for i in range(10):
     x=func_modified_landsat(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes_red,dep_cor,xvals_train_red, yvals_train,xvals_test_red, yvals_test,reduction=False)
     if x[0] > 0.4:
         acc=acc+x[0]
         c=c+1
    s.close()
    return([acc/c,len(sensor_sizes_red),v])
  else:
    s.close()
    return([testacc,len(sensor_sizes)])



#%%
def func_modified_CV(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals_train, yvals_train,xvals_test, yvals_test, reduction):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split

  #xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,random_state=None, test_size=0.2,  shuffle=True)
                                                                     
  starter_learning_rate = 0.001
  num_features=sum(sensor_sizes)
  nrow=len(yvals_train)
  num_output=num_classes

  input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
  input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

  s=tf.compat.v1.InteractiveSession()
  ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
  weights_0 = tf.Variable(tf.random.normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
  bias_0 = tf.Variable(tf.random.normal([num_layers_0]))
  weights_1 = tf.Variable(tf.random.normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
  bias_1 = tf.Variable(tf.random.normal([num_output]))

  ## Initializing weigths and biases
  hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
  predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

  ##calculate penalty terms
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  
  cumsum =[0]+ list(series.cumsum())
  
  
  loss = tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) 

  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  ## Metrics definition
  correct_prediction = tf.equal(tf.argmax(yvals_train,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  training_accuracy = []
  training_loss = []

  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:xvals_train, 
                                                         input_y: yvals_train}))
    training_loss.append(s.run(loss, {input_X: xvals_train, 
                                      input_y: yvals_train}))
  print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch]
                                                                   ))
    
  
    
  y_pred = np.rint(s.run(predicted_y, feed_dict={input_X: xvals_test}))

  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))
  
  s.close()
  return(testacc)

#%%

#IRIS-1
iris=pd.read_csv('/Users/aytijhyasaha/Downloads/datasets/sensor-selection-datasets/Iris.csv')
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(iris['Species'].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]


for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].astype(np.float32)

#writer = pd.ExcelWriter('output_1_iris1_new_tuning_10_times.xlsx')

for i in [10]:
  for j in [0]:
      result = []
      for k in range(1):
          print(i,j,k)
          x = func_modified(i, j, 10, 500, 10, 3, [1,1,1,1], dep_cor, xvals, yvals, True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_iris1 = pd.DataFrame(result)
      result_iris1.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_iris_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_iris1.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_iris.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()     
#%%
#IRIS-2

f1=iris['SepalLengthCm']
f2=iris['SepalWidthCm']
f3=iris['PetalLengthCm']
f4=iris['PetalWidthCm']
e1=f1+np.random.normal(loc=0.0, scale=0.05, size=150)
e2=f3+np.random.normal(loc=0.0, scale=0.05, size=150)
e3=f4+np.random.normal(loc=0.0, scale=0.05, size=150)
xvals=pd.concat([f1,f2,e1,f3,f4,e2,e3], axis=1, ignore_index=True).astype(np.float32)

for i in [ 10]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i, j, 10, 500, 10, 3, [2,3,2], dep_cor, xvals, yvals, True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_iris1 = pd.DataFrame(result)
      result_iris1.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_iris2_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_iris1.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_iris2.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()

#%%
#Gas-sensor

data=pd.read_csv('//Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/GasSensor(cleaned in R).csv',header=None)
#data=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/GasSensor(cleaned in R).csv')
data.drop(0,axis=1,inplace=True)
data.drop(0,axis=0,inplace=True)
data.dropna() 
data[1]=data[1].replace(['1'],1)
data[1]=data[1].replace(['2'],2)
data[1]=data[1].replace(['3'],3)
data[1]=data[1].replace(['4'],4)
data[1]=data[1].replace(['5'],5)
data[1]=data[1].replace(['6'],6)

yvals=data[1]
data=data.astype(np.float32)
yvals=to_categorical(np.asarray(yvals.factorize()[0]))
from scipy.stats import zscore
for i in range(128):
  data.iloc[:,i+1]=zscore(data.iloc[:,i+1])

data['Id']=list(range(0,13910))

shuffled = data.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 2000, 6,
                                list(repeat(8,16)),dep_cor,
                                trn.iloc[:,1:129],yvals[trn['Id']],
                                tst.iloc[:,1:129],yvals[tst['Id']],
                                reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

for i in [50]:
  for j in [ 0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500, 2000, 6,
                                  list(repeat(8,16)),dep_cor,
                                  data.iloc[:,1:129],
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_gs = pd.DataFrame(result)
      result_gs.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_gs_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_gs.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_gs.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()
      
#%%

#LRS
lrs=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/lrs_cleaned.csv')
lrs.drop('Unnamed: 0',axis=1,inplace=True)

yvals = lrs.iloc[:,1]
yvals=(yvals-yvals % 10)/10
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

from scipy.stats import zscore
for i in range(93):
  lrs.iloc[:,i+10]=zscore(lrs.iloc[:,i+10])
xvals = lrs.iloc[:,10:103]

lrs['Id']=list(range(0,len(yvals)))

shuffled = lrs.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 100,10,[44,49],dep_cor,
                                trn.iloc[:,10:103],yvals[trn['Id']],
                                tst.iloc[:,10:103],yvals[tst['Id']],
                                reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

for i in [50]:
  for j in [5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500,50,10,[44,49],dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_lrs = pd.DataFrame(result)
      result_lrs.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_lrs_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_lrs.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_lrs.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()

#%%
#RSData-1

rs=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/rs_8cl.csv')
#rs=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/sensor-selection-datasets/rs_8cl.csv')
rs.drop('Unnamed: 0',axis=1,inplace=True)

from scipy.stats import zscore
for i in range(7):
  rs.iloc[:,i]=zscore(rs.iloc[:,i])

s=[]
for i in range(8):
    s.append(rs.loc[rs.iloc[:,7]==i+1].sample(frac=1).iloc[0:200,])
    
trn_data=pd.concat(s)

xvals_train=trn_data.iloc[:,0:7]
yvals_train=trn_data.iloc[:,7]


yvals_train=to_categorical(np.asarray(yvals_train.factorize()[0]))


trn_data['Id']=list(range(0,1600))

shuffled = trn_data.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat

for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500,200,8,list(repeat(1,7)),dep_cor,trn.iloc[:,0:7],yvals_train[trn['Id']],tst.iloc[:,0:7],yvals_train[tst['Id']],False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]
    
xvals = rs.iloc[:,1:8]
yvals = rs.iloc[:,8]
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

for i in [0,20,50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500,200,8,list(repeat(1,7)),dep_cor,rs,7,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_lrs = pd.DataFrame(result)
      
      result_lrs.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_lrs_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_lrs.to_excel(writer)
      # save the excel
      writer.save()

#%%
#BCI-IV-Dataset-1
import scipy.io
from itertools import repeat

bci=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1a.mat')

xvals=pd.DataFrame(bci['cnt'])
pos=bci['mrk']['pos'][0][0][0]
y=bci['mrk']['y'][0][0][0]
yvals=[]
id=[]
for i in range(len(y)):
    id.extend(list(np.arange(pos[i]+50, pos[i]+300)))
    yvals.extend(repeat(y[i],250))
    
xvals=xvals.iloc[id]

from scipy.stats import zscore
for i in range(59):
  xvals.iloc[:,i]=zscore(xvals.iloc[:,i])
 
yvals=pd.Series(yvals)
yvals=to_categorical(np.asarray(yvals.factorize()[0]))



xvals['Id']=list(range(len(yvals)))

shuffled = xvals.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(8, 21, 2)
acc=[]


for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 10000, 2,
                                list(repeat(1,59)),dep_cor,
                                trn.iloc[:,0:59],yvals[trn['Id']],
                                tst.iloc[:,0:59],yvals[tst['Id']],
                                reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]
xvals.drop('Id',axis=1,inplace=True)

for i in [20,50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500,10000,2,list(repeat(1,59)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_bci = pd.DataFrame(result)
      result_bci.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_bci_iv_1_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_bci.to_excel(writer)
      # save the excel
      writer.save()
      
#%%
import scipy.io
from itertools import repeat

bcia=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1a.mat')
bcib=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1b.mat')
bcif=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1f.mat')
bcig=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1g.mat')

xvals=[]

xvals.append(pd.DataFrame(bcia['cnt']))
xvals.append(pd.DataFrame(bcib['cnt']))
xvals.append(pd.DataFrame(bcif['cnt']))
xvals.append(pd.DataFrame(bcig['cnt']))

pos=[]
pos.append(bcia['mrk']['pos'][0][0][0])
pos.append(bcib['mrk']['pos'][0][0][0])
pos.append(bcif['mrk']['pos'][0][0][0])
pos.append(bcig['mrk']['pos'][0][0][0])

y=[]
y.append(bcia['mrk']['y'][0][0][0])
y.append(bcib['mrk']['y'][0][0][0])
y.append(bcif['mrk']['y'][0][0][0])
y.append(bcig['mrk']['y'][0][0][0])

yvals=[]
id=[]
baseindex=[]
for i in range(4):
  baseindex.append([])
  for j in range(len(y[i])):
    id.extend(list(np.arange(pos[i][j]+50, pos[i][j]+300)))
    baseindex[i].extend(list(np.arange(pos[i][j]+401, pos[i][j]+800)))
    yvals.extend(repeat(y[i][j],250))

base=[]
for i in range(4):
    base.append(xvals[i].iloc[baseindex[i],].apply(sum,axis=0) )
    for j in range(59):
        xvals[i].iloc[:,j]=xvals[i].iloc[:,j]-base[i][j]
        
xvals=pd.concat(xvals).iloc[id]

from scipy.stats import zscore
for i in range(59):
  xvals.iloc[:,i]=zscore(xvals.iloc[:,i])
 
yvals=pd.Series(yvals)
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

for i in [20,50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500,10000,2,list(repeat(1,59)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_bci = pd.DataFrame(result)
      result_bci.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_bci_iv_1_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_bci.to_excel(writer)
      # save the excel
      writer.save()
      
#%%
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io
from itertools import repeat
from scipy.integrate import simps

bcia=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1a.mat')
xvals=pd.DataFrame(bcia['cnt'])
l=[bcia['nfo']['clab'][0][0][0][i][0] for i in range(0,59)]
P=['AF3','AF4','Fz','FC5','FC1','FC2','FC6','C3','C4','CP3','CPz','CP6','P3', 'Pz', 'P4']
xvals=xvals.iloc[:,[l.index(P[i]) for i in range(len(P))]]

pos=bcia['mrk']['pos'][0][0][0]
y=bcia['mrk']['y'][0][0][0]


id=[]
for j in range(len(y)):
    id.extend(list(np.arange(pos[j]+50, pos[j]+300)))
y=pd.Series(y)  
yvals=to_categorical(np.asarray(y.factorize()[0]))

X=[]
for i in range(200):
    X.append(xvals.iloc[id[250*i:250*(i+1)],].mul(0.1))

def bandpower(data,low,high,sf,win):
    freqs, psd = signal.welch(data, sf, nperseg=win)
    idx = np.logical_and(freqs >= low, freqs <= high)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 2.5 = 0.4
    #print(psd)
    # Compute the absolute power by approximating the area under the curve
    power = simps(psd[idx], dx=freq_res)
    return power
    
data=[]
for i in range(200):
    d=[]
    for j in range(len(P)):
        d.extend([bandpower(X[i].iloc[:,j],1,4,100,200),
           bandpower(X[i].iloc[:,j],4,8,100,250),
           bandpower(X[i].iloc[:,j],8,12,100,250),
           bandpower(X[i].iloc[:,j],13,20,100,250),
           bandpower(X[i].iloc[:,j],20,30,100,250),
           bandpower(X[i].iloc[:,j],30,50,100,250)])
    data.append(d)

data=pd.DataFrame(data)

from scipy.stats import zscore
for i in range(len(data.T)):
  data.iloc[:,i]=zscore(data.iloc[:,i])
  
data['Id']=list(range(0,len(yvals)))

shuffled = data.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 15, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 10,2,list(repeat(6,59)),dep_cor,
                                trn.iloc[:,0:6*59],yvals[trn['Id']],
                                tst.iloc[:,0:6*59],yvals[tst['Id']],
                                reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]
data.drop('Id',axis=1,inplace=True) 
result = []
for i in [0,2,5]:
  for j in [0, 2, 5]:
      for k in range(1):
          print(i,j,k)
          x = func_modified(i,j,nodes,500,10,2,list(repeat(6,15)),dep_cor,
                                  data,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
result_bci = pd.DataFrame(result)
result_bci.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
writer = pd.ExcelWriter('output_1_bci_iv_1_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
  # write dataframe to excel
result_bci.to_excel(writer)
  # save the excel
writer.save()

#%%
#WBC

wbc=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/feature selection data/breast-cancer-wisconsin-original-data')
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(wbc.iloc[:,1].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals=wbc.iloc[:,2:11]


for i in [20,50]:
  for j in [0,2,5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,10, 500, 100, 2,
                                  list(repeat(1,9)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_wbc = pd.DataFrame(result)
      result_wbc.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_wbc_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_wbc.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_wbc.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()

#%%
#wine
data=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/feature selection data/wine.data', header=None)
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(data.iloc[:,0].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals=data.iloc[:,1:14]
from scipy.stats import zscore
for i in range(13):
  xvals.iloc[:,i]=zscore(xvals.iloc[:,i])


for i in [0,50]:
  for j in [0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,10, 500, 50, 3,
                                  list(repeat(1,13)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_wine = pd.DataFrame(result)
      result_wine.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_wine_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_wine.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_wine.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()


#%%
#sonar
data=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/feature selection data/sonar.all-data', header=None)
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(data.iloc[:,60].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals=data.iloc[:,0:60]
from scipy.stats import zscore
for i in range(60):
  xvals.iloc[:,i]=zscore(xvals.iloc[:,i])


for i in [50]:
  for j in [0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,10, 500, 20, 2,
                                  list(repeat(1,60)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_sonar = pd.DataFrame(result)
      result_sonar.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_sonar_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_sonar.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_sonar.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()


#%%

data=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/feature selection data/ecoli.csv')
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(data.iloc[:,9].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_CV(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals=data.iloc[:,2:9]
from scipy.stats import zscore
for i in range(7):
  xvals.iloc[:,i]=zscore(xvals.iloc[:,i])


for i in [0,20,50]:
  for j in [0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,10, 500, 20, 8,
                                  list(repeat(1,7)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_ecoli = pd.DataFrame(result)
      result_ecoli.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_ecoli_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_ecoli.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_ecoli.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()


#%%
#mnist
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()



data1=[train_X[i].transpose().reshape(784,order="C") for i in range(60000)]
data2=[test_X[i].transpose().reshape(784,order="C") for i in range(10000)]
xvals=np.concatenate((data1,data2),axis=0)
xvals=pd.DataFrame(xvals)
from scipy.stats import zscore
for i in range(784):
  if max(xvals.iloc[:,i].min(), xvals.iloc[:,i].max(), key=abs)>0:
      xvals.iloc[:,i]=zscore(xvals.iloc[:,i])

yvals=np.concatenate((train_y,test_y),axis=0)
yvals=pd.Series(yvals)
yvals=to_categorical(np.asarray(yvals.factorize()[0]))
from itertools import repeat
for i in [20,50]:
  for j in [0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,16, 400, 30000, 10,
                                  list(repeat(28,28)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_mnist = pd.DataFrame(result)
      result_mnist.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_mnist_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_mnist.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_mnist.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()

#%%
data=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/feature selection data/SRBCT.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
yvals=data.iloc[:,2308]
yvals=to_categorical(np.asarray(yvals.factorize()[0]))
xvals=data.iloc[:,0:2308]
from scipy.stats import zscore
c=[]
for i in range(2308):
    if max(xvals.iloc[:,i].min(), xvals.iloc[:,i].max(), key=abs)>0:
        xvals.iloc[:,i]=zscore(xvals.iloc[:,i])
    else:
        c.append(i)
        

from itertools import repeat
for i in [20,50]:
  for j in [0]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,16, 400, 30, 4,
                                  list(repeat(1,2308)),dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_srbct = pd.DataFrame(result)
      result_srbct.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_2_srbct_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_srbct.to_excel(writer)
      # save the excel
      writer.save()
writer2 = pd.ExcelWriter('Dependency_matrix_srbct.xlsx')
# write dataframe to excel
pd.DataFrame(x[4]).to_excel(writer2)
# save the excel
writer2.save()