# DANN(Domain Adversarial Neural Network)
Implementation of [DANN](https://arxiv.org/abs/1505.07818) in Tensorflow 2.0 environment, written for Deepest season 6 recruiting assignment.

## Project Requirements
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

[@tf.custom_gradient](https://www.tensorflow.org/api_docs/python/tf/custom_gradient) decorator defines a tf operator with custom gradient. **GradientReversalOperator** serves as an identity operator in forward pass and a gradient reversal operator in backward pass. **GradientReversalLayer** is a child of keras layer class that wraps the custom operator.

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
Implemented in [/src/models.py](https://github.com/Joovvhan/dann_tf2.0/blob/master/src/models.py) as class variables of **DANN_Model** class.


### 3. Resizing and normalizing input data

### 4. Prepare at least 2 datasets
3 datasets are used to demonstrate implemented DANN network.
[MNIST](http://yann.lecun.com/exdb/mnist/)
[SVHN](http://ufldl.stanford.edu/housenumbers/)
[SynNumbers](https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view)

### 5. Target classification accuracy above 70%

### *. Comparison between source-only and DANN model

## Reference
[openTSNE](https://github.com/pavlin-policar/openTSNE)
