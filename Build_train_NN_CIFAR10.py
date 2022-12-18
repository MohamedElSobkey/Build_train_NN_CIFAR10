# import Libraries
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt




# Load the CIFAR-10 data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train) , (x_test,y_test) = cifar10.load_data()

 
# Pre-process the data and Explore the date

# Pre-process  the data : convert pixels intensities to double values between 0 and 1
x_train , x_test = x_train/255.0  , x_test/255.0


# check the data has the correct shape/dimension
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Plot the first 10 images from the training set and  display the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range (10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[10])
    plt.xlabel(class_names[y_train [i][0]])
    
plt.show()

# Build the tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape= (32,32,3)), #convert the images to 1-dimension array
    tf.keras.layers.Dense(10),# the last layer should be the size of the output 
    
    ])


# Once the model is built : you can call it's summary
model.summary()

# pass 1 training data image to the model and convert the predictions to numpy array
predictions = model (x_train[:1]).numpy()

print(predictions)


# Use tf.nn.softmax function to convert these logits into "probabilities" for each class 

tf.nn.softmax(predictions).numpy()

# choose an optimizer and loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

loss_fn(y_train[:1], predictions).numpy()


# Ready to compile
model.compile(optimizer='adam', loss = loss_fn, metrics = ['accuracy'])

# model.fit
model.fit(x_train, y_train, epochs = 20)

# Evaluate a model
test_loss , test_acc = model.evaluate(x_test, y_test, verbose = 2)

print('\n Test accuracy' , test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test[:5]) #test the first 5 images 

#print(predictions.shape)
predictions[0]


# Apply label and compare with the test label
print(np.argmax(predictions, axis = 1))
print(y_test[:5])


# View the first 5 images to check the validity of the labels
for i in range (5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    plt.xlabel(class_names[y_test [i][0]])
    
plt.show()


# Take a look at the learned paramaters
filters, biases = model.layers[1].get_weights()
f_min , f_max = filters.min(), filters.max()
filters = (filters - f_min)/  (f_max - f_min)
print(filters.shape)


# reshape to rgb shape
filters_rgb = filters.reshape(32,32,3,10)

#Plot the 10 filters
n_filters = 10
for i in range (n_filters):
    # get the filter
    f = filters_rgb[:, : ,: , i]
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(class_names[i])
    plt.imshow(f)
        
plt.show()





    