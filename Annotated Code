import tensorflow as tf #contructs and trains NN
import tensorflow_datasets as tfds #dataset access
import matplotlib.pyplot as plt #allows me to create visualizations

# TL;DR Load the malaria dataset
ds, info = tfds.load('malaria', split= 'train', shuffle_files=True, with_info=True)
#two varibles, ds and info = (
#both carry following: 'malaria':dataset ->
#split='train': portions dataset to see what the model will learn from; can be split into 70%, or 30% ->
#shuffle_files: shuffles files to avoid bias and ensure randomness ->
#with_info: supplies additional important information from the og dataset, like CLASS NAMES)

# TL;DR Define the input shape and number of classes
input_shape = (64, 64, 3)
num_classes = info.features['label'].num_classes
#input_shape: processes all data images into 64x64 pixels with three color channels -- resizing to digestable shapes for NN
#num_classes: accesses features of dataset via info.features, accesses info relevant to 'labels', and retrieves num of classes via .num_classes from 'labels' -- uninfected or infected == 2 classes

# TL;DR Preprocess and prepare the dataset
def preprocess(example):
    image = tf.cast(example['image'], tf.float32) / 255.0  # Normalize image
    image = tf.image.resize(image, input_shape[:2])  # Resize image
    label = tf.one_hot(example['label'], num_classes)  # One-hot encode label
    return image, label
#Image(normalizing): Different pixels in the input data have different pixel value ranges based on things like lighting -- converts og datatype to the tf.float32 datatype ("casts")
      # and divides by 255 to get values between 0 and 1, thereby "normalizing" or homogenizing datatype values for digestability throughout NN
#Image(resizing): Newly normalized image is then resized (width x height) based on first two elements in (64, 64, 3) tuple -- converts input size to 64x64 pixels

ds = ds.map(preprocess)
ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
#ds: uses .map() function to apply "preprocess" function to every single element in this dataset
#ds: .shuffle(1024) shuffles 1024 elements, .batch(32) divides dataset in batches of 32---helps process multiple examples simulatenously,
        #.prefetch(tf.data.AUTOTUNE) is preselecting/preloading batches of data so that model does not have to wait for next batch to be fetched because TENSORFLOW is concurrently loading batches in background
        #    • amount that is preloaded is based on system resources, CPU + GPU

#TL;DR Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)), #downsampling to reduce size of each image into non-overlapping region (2x2) for managability
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Sequential Models: A linear pipeline where data flows from one layer to another, each layer consisting of "nodes" that perform mathematical operations on the data
# ^^ In this example: Input -> [Convolutional Layer] -> [MaxPooling Layer] -> [Convolutional Layer] -> [MaxPooling Layer] -> [Flatten Layer] -> [Fully Connected Layers] -> [Output Layer] -> Output
#     Convolutional Layer: Powerful  image recog. tool that performs mathmematical operations called "convolutions" in "filters" that ultimately aid in extracting details such as texture, edges, or shapes at different places
#           • (32, (3, 3), activation='relu': 32 filters, each with a 3x3 size ->
#               • w = weight = learnable parameter that determines the impact a specific pixel will have on output
#               • each filter looks like this:  w1  w2  w3  and is responsible for detecting a certain feature or pattern in the input
#                                               w4  w5  w6
#                                               w7  w8  w9 -> each filter slides over input image and does math
#           • ReLU: introduces nonlinearity to the convolutional layer, which defines relationships between patterns that cannot be explained by linear functions; represents more complex patterns
#               • In ReLU, if an node recieves a negative or zero input, then ReLU blocks signal and outputs zero, essentially eliminating information that does not contribute to overall pattern recognition
#
#     MaxPooling Layer: Used for retaining important features within the imput via two main methods:
#           • Downsampling: Reducing size or resolution of input data to make the input more computationally efficient
#           • Feature Selection: Returns the maximum value in each non-overlapping pooling window (in this case 2x2) and disgards non-maximum values
#                       • Maximum values == Strongest or most prominent feature in a given region
#                       • MaxPooling Layer translates this: [[1,2,3,4], -> into 2x2 regions -> [[1,2] | [[3,4] | [[9,10]  | [[11,12],
#                                                            [5,6,7,8],                        [5,6]] | [7,8]] | [13,14]] | [15,16]] -> Selects maximum values in all regions: 6,8,14,16
#                                                            [9,10,11,12],                                        ↓
#                                                            [13,14,15,16] -> New Feature map looks like this: [[6,8] -> Downsampled result! Reduction of size!
#                                                                                                              [14,16]]
#
#     Flatten Layer: Multi-dimensional, organized feature maps and arranges them into linear list that can be fed and understood by the next layers in the network
#
#     Fully Connected/Dense Layer: Layer that is fully connected to the previous layer, meaning, in this case, 64 neuron recieves input from [ALL] previous neurons in the layer, unlike the Convoolutional Layer
#           • ReLU is applied to this layer to introduce non-linearity and find complex patterns within the input
#                        • Each of the 64 layers produces one numerical value, representing the "activation levels" of each of the neurons to a paticular input (this depends on learned weights and biases of the neurons)

#
#     Output Layer/Final Dense Layer: Layer responsible for producing the probability of each class by making predications based on learned patterns from the input data
#            • (num_classes) represents the amount of neurons in the final layer, in this case two classes/catagories that inputs will be sorted into based on probabilities
#            • activation = 'softmax': statistical operation that converts the dense layer values into valid probability distributions to ensure values fall in between 0 and 1

#TL;DR Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.compile(): function used to cofigure the training process of the neural network; essentially, the setting menu of this entire model
#      • optimizer='': minimizes loss function by adjusting weights and biases based on gradient of the loss function to optimize model performance via backpropagation
#      • loss='': mathmatical functional that measures the discreepency of a predicted output against a target output, used to gauge and optimize performance of the model
#      • metrics='': evaluation metrics used to monitor a models performance during the training phase; 'accuracy' measures percentage of correctly classified samples out of the total dataset (%)

#TL;DR Train the model
model.fit(ds, epochs=5)
#      • model: my neural network
#      • .fit(ds, epochs=5): trains my neural network on the preprocessed tensorflow dataset 5 times during the training

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

test_ds = tfds.load('malaria', split='train', shuffle_files=True)

test_ds = test_ds.map(preprocess)

num_images = 5
test_images = []
test_labels = []
predicted_labels = []

for i, (image, label) in enumerate(test_ds.take(num_images)):
    test_images.append(image)
    test_labels.append(label)

    predictions = model.predict(tf.expand_dims(image, axis=0))
    predicted_class = tf.argmax(predictions, axis=1)
    predicted_labels.append(predicted_class.numpy()[0])

class_names = info.features['label'].names
test_labels = [tf.argmax(label) for label in test_labels]

fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

for i, ax in enumerate(axes):
    ax.imshow(test_images[i])
    ax.set_title(f'True: {class_names[test_labels[i]]}\nPredicted: {class_names[predicted_labels[i]]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
