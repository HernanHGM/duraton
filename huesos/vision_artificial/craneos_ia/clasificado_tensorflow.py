# %% IMPORT LIBRARIES
import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import cohen_kappa_score
# %% IMPORT DATA
train_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\train'
validation_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\validation'

train_aguila_real_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\train\\aguila_real'
train_aguila_imperial_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\train\\aguila_imperial'
validation_aguila_real_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\validation\\aguila_real'
validation_aguila_imperial_dir = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\validation\\aguila_imperial'

print(f"There are {len(os.listdir(train_aguila_real_dir))} images of aguila_real for training.\n")
print(f"There are {len(os.listdir(train_aguila_imperial_dir))} images of aguila_imperial for training.\n")
print(f"There are {len(os.listdir(validation_aguila_real_dir))} images of aguila_real for validation.\n")
print(f"There are {len(os.listdir(validation_aguila_imperial_dir))} images of aguila_imperial for validation.\n")

# %% Look some images

# print("Sample aguila_real image:")
# plt.figure()
# plt.imshow(load_img(f"{os.path.join(train_aguila_real_dir, os.listdir(train_aguila_real_dir)[0])}"))
# plt.show()

# print("\nSample aguila_imperial image:")
# plt.figure()
# plt.imshow(load_img(f"{os.path.join(train_aguila_imperial_dir, os.listdir(train_aguila_imperial_dir)[0])}"))
# plt.show()

# %% CHECK IMAGE SIZE
# Load the first example of a horse

sample_image  = load_img(f"{os.path.join(train_aguila_real_dir, os.listdir(train_aguila_real_dir)[0])}")

# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)

print(f"Each image has shape: {sample_array.shape}")

# %% IMAGE GENERATOR

# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator: tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class 
  # Don't forget to normalize pixel values and set arguments to augment the images 
  train_datagen = ImageDataGenerator(rescale=1.0/255.,
                                     rotation_range=45,
                                     width_shift_range=0.1,
                                     height_shift_range=0.3,
                                     shear_range=0.1,
                                     zoom_range=0.2,
                                     # brightness_range=(0.5,0.9),
                                     channel_shift_range=10.0,
                                     horizontal_flip=True,
                                     fill_mode='constant')

  # Pass in the appropriate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=32,
                                                      class_mode='binary',
                                                       # color_mode='grayscale',
                                                      target_size=(300, 300))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  # Remember that validation data should not be augmented
  validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

  # Pass in the appropriate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=32, 
                                                                class_mode='binary',
                                                                # color_mode='grayscale',
                                                                target_size=(300, 300))
  ### END CODE HERE
  return train_generator, validation_generator
# %% CHECK FILES

train_generator, validation_generator = train_val_generators(train_dir, validation_dir)

# %% IMPORT INCEPTION
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = 'E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\resultados\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# %% CREATE PRETRAINED MODEL
def create_pre_trained_model(local_weights_file):
  """
  Initializes an InceptionV3 model.
  
  Args:
    local_weights_file (string): path pointing to a pretrained weights H5 file
    
  Returns:
    pre_trained_model: the initialized InceptionV3 model
  """
  ### START CODE HERE
  pre_trained_model = InceptionV3(input_shape = (300, 300, 3),
                                  include_top = False, 
                                  weights = None) 

  pre_trained_model.load_weights(local_weights_file)

  # Make all the layers in the pre-trained model non-trainable
  for layer in pre_trained_model.layers[:279]: #228=mixed7; 279=mixed9
     layer.trainable = False

  ### END CODE HERE

  return pre_trained_model
# %% CHECK PRETRAINED MODEL
pre_trained_model = create_pre_trained_model(local_weights_file)

# Print the model summary
pre_trained_model.summary()
# i = 0
# for layer in pre_trained_model.layers: #228=mixed7
#    print(layer.name, i)
#    i=i+1
# %% CALLBACK

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True
# %% ULTIMA CAPA DE INCEPTION
def output_of_last_layer(pre_trained_model):
  """
  Gets the last layer output of a model
  
  Args:
    pre_trained_model (tf.keras Model): model to get the last layer output from
    
  Returns:
    last_output: output of the model's last layer 
  """
  ### START CODE HERE
  last_desired_layer = pre_trained_model.get_layer('mixed10')
  print('last layer output shape: ', last_desired_layer.output_shape)
  last_output = last_desired_layer.output
  print('last layer output: ', last_output)
  ### END CODE HERE

  return last_output

# GRADED FUNCTION: create_final_model
def create_final_model(pre_trained_model, last_output):
  """
  Appends a custom model to a pre-trained model
  
  Args:
    pre_trained_model (tf.keras Model): model that will accept the train/test inputs
    last_output (tensor): last layer output of the pre-trained model
    
  Returns:
    model: the combined model
  """
  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(last_output)

  ### START CODE HERE

  # Add a fully connected layer with 1024 hidden units and ReLU activation
  x = layers.Dense(1024, activation = 'relu')(x)
  # Add a dropout rate of 0.2
  x = layers.Dropout(0.5)(x)  
  # Add a final sigmoid layer for classification
  x = layers.Dense(1, activation='sigmoid')(x)        

  # Create the complete model by using the Model class
  model = Model(inputs=pre_trained_model.input, outputs=x)

  # Compile the model
  model.compile(optimizer = Adam(learning_rate=0.0001), 
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

  ### END CODE HERE
  
  return model
# %% DEFINO MODELO FINAL
last_output = output_of_last_layer(pre_trained_model)
# Print the type of the pre-trained model
print(f"The pretrained model has type: {type(pre_trained_model)}")
# Save your model in a variable
model = create_final_model(pre_trained_model, last_output)

# Inspect parameters
total_params = model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")

# %% ENTRENAMIENTO

callbacks = myCallback()

history = model.fit(train_generator,
                    validation_data = validation_generator,
                    epochs = 190,
                    verbose = 2,
                    callbacks=callbacks)
                    # ,
                    # class_weight={ 0 : 0.414 , 1 : 0.586 },
                    # callbacks=callbacks)
# %% NEW MODEL
def create_model():

  ### START CODE HERE       

  # Define the model
  # Use no more than 2 Conv2D and 2 MaxPooling2D
  model = tf.keras.models.Sequential([ 
    # Note the input shape is the desired size of the image 28x28 with 1 bytes color
    # This is the first convolution
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  
  model.compile(optimizer = Adam(learning_rate=0.0001), 
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
  ### END CODE HERE       
  
  return model

# Save your model
model = create_model()

# Train your model
history = model.fit(train_generator,
                    epochs=120,
                    validation_data=validation_generator)

# %% SAVE WEIGTHS
from tensorflow.keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="modelo8.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)
# %%
# Save the weights


# ! mkdir -p saved_model
# model.save('E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\model_m7_pretrained_balanced.h5')

# model.save('E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\model_m10_notrained_nobalanced.h5')
model.save('E:\\trabajo_pajaros\\huesos\\vision_artificial\\craneos_ia\\yoquese.h5')



# Evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# %% Plot the training and validation accuracies for each epoch

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)























