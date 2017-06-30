import numpy as np #imports numpy as np. Numpy is used for basic matrix multiplication, addition, subtraction etc.
import tensorflow as tf #imports tensorflow as tf. Tensorflow is an n-dimensional matrix, just like a 1-D vector, 2-D array, 3-D array etc.

from tensorflow.examples.tutorials.mnist import input_data #imports mnist input data from tensorflow examples. 
#Mnist data set consists of images of numbers from 0-9, each image is a 28*28 dimensional. There are total 60k training images and 10k test images.
mnist = input_data.read_data_sets("MNIST/data/", one_hot=True) #using input data call read data sets from a folder MNIST/data and store in mnist.
#one hot vector is used which means at once only one class will be true. Since our images have labels 0-9 that means out of all 10 classes only 1 class will be true at a time rest all will be zero.

Xtr , Ytr = mnist.train.next_batch(5000) #we use 50K training images and assign the images to Xtr and respective labels of images to Ytr.
Xte , Yte = mnist.test.next_batch(1000) # we use 10K test images and assign the images to Xte and respective test label of images to Yte.

#placeholder is like a variable to which we will assign data later on. It will allow us to do operations and build our computation graph without feeding in data.
#xtr will hold the training images in form of matrix,the dimensions of xtr will be in our case 5000*784, that is why we use None which allows us to vary the dimensionality of our rows.
#we use float to define its type.
xtr = tf.placeholder("float",[None,784]) 
#similarly xte will hold the test images in form matrix which is squashed in to a 784-D column vector.
xte = tf.placeholder("float",[784])

#nearest neighbour calculation and we use L1 manhattan distance to calculate. 
#In Knn we subtract all the training images with a single test image at a time and calculate the minimum distance between them. The image which has minimum distance we then predict its respective class.
#reduction indices =1 will sum all the values in matrix across all columns.
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))),reduction_indices=1)

#arg_min will return us the index of the value which is minimum in the matrix across all rows. 
pred = tf.arg_min(distance,0)

#we initialise accuracy as zero.
accuracy = 0.

#here we initialise all the variables in our model.
init = tf.global_variables_initializer()





#this is a class that runs all the tensorflow operations and launches the graph in a session. All the operations have to be within the indentation. 
with tf.Session() as sess:
    sess.run(init) #sess.run(init), runs the variables that were initialised in the previous step and evaluates the tensor 
    
#we use for loop, to loop around all the 1000 test images and for each test image and all training images we call pred which returns the index of minimum value.    
#using print function we print all test cases, label of index of the minimum value and the label of the actual test image.
    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:]})
        print "Test", i , "Prediction", np.argmax(Ytr[nn_index]), "Ground Truth:", np.argmax(Yte[i])
    
#we then compare the predicted lablel with the actual test label and for all correctly predicted labels we calculate accuracy based on the length of the test data set.    
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte) 
    
    print "Done!"
    print "Accuracy:", accuracy #finally we print the accuracy.