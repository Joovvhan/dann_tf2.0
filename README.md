# DANN(Domain Adversarial Neural Network)
Implementation of [DANN](https://arxiv.org/abs/1505.07818) in Tensorflow 2.0 environment, written for Deepest season 6 recruiting assignment.

## Assignment Requirements
### 1. Implementation of Gradient Reversal Layer(GRL)

``` python
@tf.custom_gradient
def GradientReversalOperator(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversalOperator(inputs)
```
Implemented in [/src/models.py](https://github.com/Joovvhan/dann_tf2.0/blob/master/src/models.py)

[@tf.custom_gradient](https://www.tensorflow.org/api_docs/python/tf/custom_gradient) decorator defines a tf operator with custom gradient.

**GradientReversalOperator** serves as an identity operator in forward pass and a gradient reversal operator in backward pass.

**GradientReversalLayer** is a child of keras layer class that wraps the custom operator.

### 2. Implementation of feature extractor, class classifier, domain classifier
``` python
self.feature_extractor = Sequential([
    Conv2D(filters=64, kernel_size=5, strides=1, kernel_regularizer=l2(0.001), padding='same', input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=3, strides=2),
    ...
    Flatten()            
])

self.label_predictor = Sequential([
    Dense(3072, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    ...
    Dense(10, activation='softmax')
])

self.domain_classifier = Sequential([
    GradientReversalLayer(),
    Dense(1024, kernel_regularizer=l2(0.001)),
    ...
    Dense(2, kernel_regularizer=l2(0.001)),
    Activation('softmax')          
])		

self.predict_label = Sequential([
    self.feature_extractor,
    self.label_predictor
])

self.classify_domain = Sequential([
    self.feature_extractor,
    self.domain_classifier
])
```
Implemented in [/src/models.py](https://github.com/Joovvhan/dann_tf2.0/blob/master/src/models.py) as class variables of **DANN_Model** class using **tf.keras.Sequential**.

**feature_extractor** is a set of stacked convolutional layers with a **Flatten layer** in the end.

**label_predictor** predicts the label with stacked neural networks.

**domain_classifier** predicts the domain where the inputs comes from, 0 for a source, 1 for a target. A **GradientReversalLayer** is located at the front.

For all three architectures, both **MNIST** version and **SVHN** version from the original paper are implemented. The only difference from the original implementation is the existence of batch normalization layers.

**predict_label** concatenates the **feature_extractor** and the **label_predictor**. This path predicts labels of input images.

**classify_domain** concatenates the **feature_extractor** and the **domain_classifier**. This path predicts domains of inputs images.

### 3. Resizing and normalizing input data
#### Input Normalization
``` python
def load_data(data_category):
    ...
    x_train = x_train[:TRAIN_NUM] / 255.0
    y_train = y_train[:TRAIN_NUM]

    x_test = x_test[:TEST_NUM] / 255.0
    y_test = y_test[:TEST_NUM]

    return (x_train, y_train, x_test, y_test)
```
Implemented in [/src/preprocessing.py](https://github.com/Joovvhan/dann_tf2.0/blob/master/src/preprocessing.py)
Pixel values are devided by maximum value(255).

#### Data Resize
``` python
def data2dataset(x, y, data_category):
    if (data_category == 'MNIST'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(pad_image)
        dataset = dataset.map(duplicate_channel)
        dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))

    elif (data_category == 'SVHN' or data_category == 'SYN'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))
    return dataset
    
    def pad_image(x, y):
	paddings = tf.constant([[2, 2,], [2, 2]])
	new_x = tf.pad(x, paddings, "CONSTANT")
	return (new_x, y)

    def duplicate_channel(x, y):
	new_x = tf.stack([x, x, x], axis = -1)
	return (new_x, y)
```
Implemented in [/src/preprocessing.py](https://github.com/Joovvhan/dann_tf2.0/blob/master/src/preprocessing.py)

**data2dataset** prepares datasets by tf.data.Dataset.

MNIST dataset has image size of (28, 28, 1), and both SVHN and SynNumbers have image size of (32, 32, 3). Input image size of (32, 32, 3 ) is considered as appropriate.

**pad_image** function pads zeros around MNIST images, and **duplicate_channel** function stacks the same image by three times to create three channel image.

### 4. Prepare at least 2 datasets
As shown in section 3, 3 datasets are used to demonstrate implemented DANN network.

[MNIST](http://yann.lecun.com/exdb/mnist/)

[SVHN](http://ufldl.stanford.edu/housenumbers/)

[SynNumbers](https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view)

### 5. Target classification accuracy above 70%

### *. Comparison between source-only and DANN model



## Remarks

## References
[DANN](https://arxiv.org/abs/1505.07818)
[openTSNE](https://github.com/pavlin-policar/openTSNE)
