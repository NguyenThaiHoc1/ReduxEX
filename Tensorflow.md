Lecture 2
Talk about Near neighbor classifier 

16:21 How to compare for data training ?
Bởi vì nếu chúng ta sử dụng mình ảnh của mình đi so sánh 
thì nó sẽ mang lại rất nhiều kết quả và cho ta nhiều sự lựa chọn
Ở đây họ để xuất algorithm: L1 distance 
=> bằng việc so sánh các điểm ảnh cá nhân giống nhau
** Như chúng ta cũng đã biết 1 tấm ảnh dc mô tả bởi 1 mảng đa chiều trong đó
mỗi phần từ của nó tương ứng với 1 pixels [0-255]
L1 distance giúp chúng ta có thể do lường được sự khác biệt giữa những
tấm ảnh thông qua pixel 
 

Nhưng  L2 mới là cái người ta lựa chon nhiều 
L2 (Euclidean) distance
https://machinelearningcoban.com/2016/12/27/categories/

http://toughdev.vn/qa/345996/kho%E1%BA%A3ng-c%C3%A1ch-gi%E1%BB%AFa-numpy-m%E1%BA%A3ng-columnwise

`pip install python-mnist`


numpy 
http://viet.jnlp.org/home/nguyen-van-hai/nghien-cuu/mlearning/building-machine-learning-system-using-python/chng-1-bt-u-vi-python/mot-vai-vi-du-numpy

cac ham matlab
https://github.com/tsaith/ufldl_tutorial

import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist

np.mean(<matrix>, axis = {0, 1}): neu axis = 0 tinh trung binh theo chieu doc , axis = 1 tinh trung binh theo chieu ngang 
D = cdist(X, centers) : D la 1 matrix khoang cach so dong dua vao X So voi cac diem trong centers 
np.shape: liet ke thong tin dong cot => ( ‘so dong’, ‘so cot’)


kmeans = KMeans(n_clusters=2).fit(X)
label = kmeans.predict(X)
imagejpg = np.zeros_like(X)

for k in range(2):
    imagejpg[label == k] = kmeans.cluster_centers_[k]
    
imgShowWithK = imagejpg.reshape((imagejpg.shape[0], imagejpg.shape[1], imagejpg.shape[2]))
plt.imshow(imgShowWithK)
plt.show()




fix loi 548
anaconda-navigator --reset




cái mình nhắm tới 
https://github.com/diegocavalca/machine-learning/tree/master/supervisioned/object.detection_tensorflow
https://www.youtube.com/watch?v=_zZe27JYi8Y



tensolflow 
https://github.com/tensorflow/models/tree/master/research


cai nay can luu y <….>
https://github.com/qdraw/tensorflow-face-object-detector-tutorial
https://github.com/datitran/raccoon_dataset/blob/master/data/raccoon_labels.csv


diegocavalcakhông phải thứ cần nhưng đáng lưu ý 
https://medium.com/towards-data-science/how-to-train-a-tensorflow-face-object-detection-model-3599dcd0c26f

cai nay cung can xem
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
https://github.com/tensorflow/tensorflow


https://kipalog.com/posts/Bat-dau-voi-Machine-Learning-thong-qua-Tensorflow--Phan-I-2

SSD python
https://www.quora.com/Is-it-better-to-do-Python-development-on-an-SSD-or-on-a-mechanical-disk


xem de hieu them 
https://www.youtube.com/watch?v=QfNvhPx5Px8
tensor flow 
https://www.youtube.com/watch?v=BhpvH5DuVu8&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP&index=3


https://github.com/tensorflow/models/tree/master/research/object_detection/utils


https://github.com/tensorflow/models

I’ll take you away, each and every time.


tổ chức project 
https://github.com/davidsandberg/facenet
pip install face_recognition


https://github.com/ageitgey/face_recognition


/anaconda2/bin/python -u -c "import setuptools, tokenize;__file__='/private/var/folders/jy/5zjml4q54dv2fmdqz2xvwc3w0000gn/T/pip-build-xFHhPw/dlib/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))


https://github.com/PCJohn/FaceDetect

Mọi người cho em hỏi 
Mình  có viết 2 cái nhỏ  
1. Dùng opencv để hiện thị camera + lưu camera 
2. Face detection từ các tấm ảnh 
Thì khi mình hợp phần xử lý của 2 trong phần lấy while (phần while lấy từng bức ảnh để xử lý thì camera không hiện lên cũng như lưu video cũng hư luôn ?)
mình không biết nó bị sao @@
Mọi người ai biết hoặc từng bị qua mong dc chỉ giáo :D



dataset 
http://www.robots.ox.ac.uk/~vgg/data/hands/index.html

cai nay hay
https://github.com/yeephycho/tensorflow-face-detection
https://medium.com/towards-data-science/how-to-train-a-tensorflow-face-object-detection-model-3599dcd0c26f

https://noahingham.com/blog/facerec-python.html



print("image " + image_path.split('.')[0] + '_labeled.jpg')
            keywords = ""
            i = 0
            while (i < len(np.squeeze(scores))):
                currentScore = np.squeeze(scores)[i]
                if currentScore >= 0.75:
                    currentClasses = np.squeeze(classes).astype(np.int32)[i]
                    keywords += category_index[currentClasses]["name"] + ", "
                i = i + 1

            if len(keywords) >= 3:
                print(keywords[:-2])

            while(True):
                # co the dung cv2 de xuat ra
                cv2.imshow('Output', cv2.resize(image_np, (800, 600)))

                key = cv2.waitKey(10) & 0xFF
                if key==27:
                    break


https://ideone.com/9JKJCj
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721
https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8
https://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/
https://github.com/oarriaga/face_classification


https://www.quora.com/What-is-the-deep-neural-network-known-as-%E2%80%9CResNet-50%E2%80%9D




http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html


CNN overview
http://cs231n.github.io/convolutional-networks/#overview



code 
https://github.com/oarriaga/face_classification

https://github.com/forsythe/tensorflow-emotion-detection





