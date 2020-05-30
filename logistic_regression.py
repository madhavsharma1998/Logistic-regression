import glob
import PIL
import skimage
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as imaa

images = []


for f in glob.glob("D:/for ml/face rec/*.JPG") :
    im = (PIL.Image.open(f))
    new = im.resize((200,200))
    images.append(new)
        
    
data = []
for i in images :
    a = np.asarray(i)
    data.append(a)
    
y = [1,1,1,0,1,1,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1]
y=np.array(y)
data = np.array(data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(data,y,test_size=0.25, random_state=0)

m_x=x_train.shape[0]
print(m_x)
m_y=y_test.shape[0]
print(m_y)
m_x=x_train.shape[1] 
print(m_x)
m_x=x_train.shape[2]
print(m_x)
m_x=x_train.shape[3] 
print(m_x)

x_test = x_test.reshape(x_test.shape[0],-1).T
x_train = x_train.reshape(x_train.shape[0],-1).T

x_test = x_test/255 
x_train = x_train/255

def sigmoid(z) :
    s = 1/(1+np.exp(-z))
    return s
    
print(sigmoid(0))
# initializing with zero
def iwz(dim) :
    w= np.zeros([dim,1])
    b= 0
    return w,b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)                                    # compute activation
    cost = (-1/m)*((Y*(np.log(A)) + (1-Y)*(np.log(1-A))).sum()) 
    
    dw = (1/m)*(np.dot(X,(A-Y).T))
    db = (1/m)*((A-Y).sum())
    
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w,b,X,Y)
    
        dw = grads["dw"]
        db = grads["db"]
        
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid((np.dot(w.T,X))+b)
    
    for i in range(A.shape[1]):
        
        if(A[0][i]>0.5) :
          Y_prediction[0][i]= 1
        else:
           Y_prediction[0][i]= 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(x_train, y_train, x_test, y_test, num_itr= 2000, learning_rate= 0.05,print_cost=False):
    w,b = iwz(x_train.shape[0])
    para, grad, cost= optimize(w,b,x_train,y_train,num_itr,learning_rate,print_cost=False)
    
    w= para["w"]
    b= para["b"]
    
    Y_prediction_test = predict(w, b, x_test)
    Y_prediction_train = predict(w, b, x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
    
    d = {"costs": cost,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_itr}
    
    return d, Y_prediction_test

d,Y_prediction_test= model(x_train, y_train, x_test, y_test, num_itr= 5000, learning_rate= 0.005,print_cost=False)



costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


my_image =  PIL.Image.open("D:/for ml/imaa.JPG")  # change this to the name of your image file 

my_image = my_image.resize((200,200))

image = np.asarray(my_image)

image = image/255.
image=image.reshape(200*200*3,1)

my_predicted_image = predict(d["w"], d["b"], image)
