import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import NASNetMobile
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
import cv2
import tensorflow as tf
import seaborn as sns 
from sklearn import metrics
# from keras.callbacks import History
# history = History()


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# In[2]:
# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
# with tf.Session() as sess:
#     print (sess.run(c))

base_model=NASNetMobile(weights='imagenet',include_top=False, input_shape = (224, 224, 3), pooling = 'avg') #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
#x=GlobalAveragePooling2D()(x)
#x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
#x=Dense(1024,activation='relu')(x) #dense layer 2
#x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(7,activation='softmax')(x) #final layer with softmax activation


# # In[3]:


model=Model(inputs=base_model.input,outputs=preds)

# #specify the inputs
# #specify the outputs
# #now a model has been created based on our architecture


# # In[4]:


# for layer in base_model.layers[:20]:
#     layer.trainable=False
# for layer in base_model.layers[20:]:
#     layer.trainable=True


# # In[5]:

# #model = load_model("/media/srinath/Major Project/Major/vggkerasmajor.h5")
# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# train_generator=train_datagen.flow_from_directory('/media/srinath/Major Project/Major/normalised', # this is where you specify the path to the main data folder
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)


# # # In[33]:


# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy

# step_size_train=train_generator.n//train_generator.batch_size
# history = model.fit_generator(generator=train_generator,
#                     steps_per_epoch=step_size_train,
#                     epochs=3)


# model.save('nasnetmobilemajor.h5')
# print(history.history.epochs())

#model.summary()
#print(model.summary())




model.load_weights("/media/srinath/Major Project/Major/nasnetweights.best.hdf5")
predict = []
for i in range (1,3040):
    # img = cv2.imread("/media/srinath/Major Project/Major/testdata/%d.jpg"%i)
    # img = cv2.resize(img,(224,224))
    # img = np.reshape(img,[1,224,224,3])
    # output = model.predict(img)
    # predict.append(output.argmax())



    img = image.load_img("/media/srinath/Major Project/Major/test/%d.jpg"%i, target_size = (224,224)) #---->  KERAS WAY
    #print("%d.jpg"%i)
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image  =preprocess_input(test_image)
    #print(test_image.shape)
    output = model.predict(test_image)
    #output = output.reshape(len(output), 2, -1)
    #print(output.shape)
    #print(output)
    predict.append(output.argmax())
 

#print(predict)

four = np.full(608,0)
zero = np.full(64,1)
athul = np.concatenate((four,zero),axis = None)
two = np.full(787,2)
athul = np.concatenate((athul, two), axis = None)
six = np.full(159,3)
athul = np.concatenate((athul, six), axis = None)
five = np.full(52,4)
athul = np.concatenate((athul, five), axis = None)
three = np.full(1045,5)
athul = np.concatenate((athul, three), axis = None)
one = np.full(324,6)
athul = np.concatenate((athul, one), axis = None)

#print(athul)

cnf_matrix = metrics.confusion_matrix(athul,predict)
sns.heatmap(data=cnf_matrix, annot=True, vmin = -50, vmax = 0, cbar = False)
plt.show()

print("Accuracy:",metrics.accuracy_score(athul,predict))
print("Precision:",metrics.precision_score(athul,predict, average = None))
print("Recall or True Positive Rate:",metrics.recall_score(athul,predict, average = None))
print("F1 score:", metrics.f1_score(athul, predict, average = None))
print("Precision:",metrics.precision_score(athul,predict, average = 'weighted'))
print("Recall or True Positive Rate:",metrics.recall_score(athul,predict, average = 'weighted'))
print("F1 score:", metrics.f1_score(athul, predict, average = 'weighted'))



# def plot_history(histories, key='categorical_crossentropy'):
#   plt.figure(figsize=(16,10))
#   epoch = range(100)
#   for name, history in histories:
#     val = plt.plot(epoch, history.history['val_'+key],
#                    '--', label=name.title()+' Val')
#     plt.plot(epoch, history.history[key], color=val[0].get_color(),
#              label=name.title()+' Train')

#   plt.xlabel('Epochs')
#   plt.ylabel(key.replace('_',' ').title())
#   plt.legend()

#   plt.xlim([0,max(history.epoch)])

# print(model.metrics)
# plot_history([('baseline', model)])




#fpr, tpr, _ = metrics.roc_curve(athul,  predict)
# auc = metrics.roc_auc_score(athul, predict, average = 'weighted')
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('Receiver Operating Characteristics Curve')
# plt.show()











#model.save('/media/srinath/Major Project/Major/mobilenetmajor.h5')

# img = image.load_img("auto.jpeg", target_size = (224,224))
# test_image = image.img_to_array(img)
# test_image = np.expand_dims(test_image, axis = 0)
# test_image  =preprocess_input(test_image)
# print(test_image.shape)

# output = model.predict(test_image)
# output = output.reshape(len(output), 2, -1)
# print(output.shape)
# print(output)
# print(output.argmax())

# from sklearn import metrics
# cnf_matrix = metrics.confusion_matrix([0,0,0,1,1,1,2,2,2], [0,0,0,1,1,1,0,1,2])
# sns.heatmap(data=cnf_matrix, annot=True)
# plt.show()
# pred_bboxes  =output[...,4]*224
# pred_shapes = output[..., 4:5]

# print(pred_shapes.shape, pred_shapes)
# print(pred_bboxes.shape, pred_bboxes)


# print(output)
# print(output.argmax())

# bbox_util = BBoxUtility(6)
# results = bbox_util.detection_out(output)
# print(results)



















































#................LOADING THE IMAGE TO BE DONE BY KERAS. NOT BY OPENCV OR PIL................#
#img = cv2.imread("copy2.jpg",1)
#img = np.array("copy2.jpg")
#print(img.shape)
# plt.imshow(img)
# plt.show()
# height, width, channels = img.shape
# img = cv2.resize(img,(224,224))

# plt.imshow(img)
# plt.show()

#img = img.reshape(1, 224,224, 3)


# cv2.imshow("temp",img)
# cv2.waitKey(0)
#height, width, channels = img.shape
#print(height,width,channels)