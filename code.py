
#import các thư viện numpy và tensorflow, tương ứng
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt #import thư viện matplotlib và cho phép hiển thị đồ thị
#xoá hiển thị đầu ra trên giao diện IPython
import IPython 
from IPython.display import clear_output


print("Load MNIST Database")
mnist = tf.keras.datasets.mnist #tải dữ liệu MNIST từ thư viện keras.datasets của TensorFlow
(x_train,y_train),(x_test,y_test)= mnist.load_data() # tải dữ liệu huấn luyện và dữ liệu kiểm tra từ tập dữ liệu MNIST
x_train=np.reshape(x_train,(60000,784))/255.0 #chuẩn hóa dữ liệu huấn luyện và dữ liệu kiểm tra về khoảng giá trị từ 0 đến 1
x_test= np.reshape(x_test,(10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train]) #mã hóa one-hot vector cho nhãn của dữ liệu huấn luyện và dữ liệu kiểm tra
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
print(x_train.shape)
print(y_train.shape)

#định nghĩa hàm
def relu(x):
    return np.maximum(0,x)
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def Forwardpass(X,Wh1,bh1,Wh2,bh2,Wo,bo): #thực hiện quá trình lan truyền thuận
    zh1 = X@Wh1.T + bh1
    ah1 = relu(zh1)
    zh2 = ah1@Wh2.T + bh2
    ah2 = sigmoid(zh2)
    zo = ah2@Wo.T + bo
    o = softmax(zo)
    return o
def AccTest(label,prediction):    # tính toán độ chính xác
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy
#khai báo learning rate, số lượng epoch (vòng lặp),số lượng mẫu huấn luyện và kiểm tra,
learningRate = 0.01 
Epoch=20
NumTrainSamples=60000
NumTestSamples=10000
#khai báo số lượng đầu vào, số lượng đơn vị ẩn trong hai lớp ẩn, số lượng lớp đầu ra
NumInputs=784
NumHiddenUnits=512
NumHiddenUnits2=512
NumClasses=10

#khai báo ma trận trọng số và vector bias
#khởi tạo với giá trị ngẫu nhiên
# 1st hidden layer
Wh1=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumInputs)))
bh1= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh1= np.zeros((NumHiddenUnits,NumInputs))
dbh1= np.zeros((1,NumHiddenUnits))
# 2nd hidden layer
Wh2=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits2,NumHiddenUnits)))
bh2= np.random.uniform(0,0.5,(1,NumHiddenUnits2))
dWh2= np.zeros((NumHiddenUnits2,NumHiddenUnits))
dbh2= np.zeros((1,NumHiddenUnits2))
# Output layer
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits2))
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits2))
dbo= np.zeros((1,NumClasses))

#khởi tạo danh sách rỗng lưu giá trị mất mát và độ chính xác 
loss = []
Acc = []
Batch_size = 200 #khai báo kích thước của mỗi batch
#khởi tạo một mảng các chỉ số từ 0 đến NumTrainSamples-1 
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range (Epoch): #bắt đầu vòng lặp qua các epoch
  np.random.shuffle(Stochastic_samples) #sắp xếp ngẫu nhiên các chỉ số
  for ite in range (0,NumTrainSamples,Batch_size): #bắt đầu vòng lặp qua từng batch trong quá trình huấn luyện
    #feed fordward propagation
    Batch_samples = Stochastic_samples[ite:ite+Batch_size] #lấy một batch mẫu từ mảng Stochastic_samples
    #lấy các mẫu huấn luyện và nhãn tương ứng của batch hiện tại
    x = x_train[Batch_samples,:]
    y = y_train[Batch_samples,:]
    #tính toán giá trị 
    zh1 = x@Wh1.T + bh1         #tính toán giá trị đầu ra của lớp ẩn
    ah1 = relu(zh1)             #áp dụng hàm kích hoạt ReLU cho đầu ra của lớp ẩn
    zh2 = ah1@Wh2.T + bh1
    ah2 = sigmoid(zh2)          #áp dụng hàm kích hoạt sigmoid
    zo=ah2@Wo.T + bo
    oo = softmax(zo)            #áp dụng hàm kích hoạt softmax
    #tinh toan loss
    loss.append(-np.sum(np.multiply(y,np.log10(oo))))  #tính toán giá trị mất mát thông qua hàm cross-entropy và thêm vào danh sách loss
    
    #tinh toan loi cho ouput layer
    d = oo-y
    
    #lỗi lan truyền ngược
    dh = d@Wo #tính toán sai số truyền ngược
    dhs2 = np.multiply(np.multiply(dh,ah2),(1-ah2)) #tính toán sai số truyền ngược từ lớp ẩn thứ hai đến lớp ẩn đầu tiên sử dụng đạo hàm của hàm sigmoid
    dhs1 = np.multiply(np.multiply(np.multiply(dh,ah2),(1-ah2)),(1-ah1)) #tính toán sai số truyền ngược từ lớp ẩn đầu tiên đến đầu vào sử dụng đạo hàm của hàm ReLU
    
    #cap nhat trong so
    dWo = np.matmul(np.transpose(d),ah2) #tính toán đạo hàm của hàm mất mát theo trọng số kết nối
    dbo = np.mean(d)                     #tính toán đạo hàm của hàm mất mát theo bias của lớp đầu ra
    dWh2 = np.matmul(np.transpose(dhs2),ah1)
    dbh2 = np.mean(dhs2)  
    dWh1 = np.matmul(np.transpose(dhs1),x)
    dbh1 = np.mean(dhs1) 
    
    #cập nhật giá trị của các tham số trong mạng neural 
    Wo =Wo - learningRate*dWo/Batch_size
    bo =bo - learningRate*dbo
    Wh1 =Wh1-learningRate*dWh1/Batch_size
    bh1 =bh1-learningRate*dbh1
    Wh2 =Wh2-learningRate*dWh2/Batch_size
    bh2 =bh2-learningRate*dbh2
    #Kiểm tra độ chính xác 
    prediction = Forwardpass(x_test,Wh1,bh1,Wh2,bh2,Wo,bo) #thực hiện lan truyền thuận trên dữ liệu kiểm tra để đưa ra các dự đoán
    Acc.append(AccTest(y_test,prediction)) #tính toán độ chính xác của mô hình trên dữ liệu kiểm tra
    clear_output(wait=True)
    #vẽ đồ thị mất mát
    plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')
    plt.show()
    #in kết quả
    print('Epoch:', ep )
    print('Accuracy:',AccTest(y_test,prediction) )
    print('Loss:',-np.sum(np.multiply(y,np.log10(oo))) )