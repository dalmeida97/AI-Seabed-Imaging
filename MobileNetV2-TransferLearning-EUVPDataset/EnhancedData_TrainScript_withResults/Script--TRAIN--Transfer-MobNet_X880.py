import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to folders and annotations (csv files)
train_folder = 'Train_images880'                     # Chose either original or enhanced images for training
augmented_train_folder = 'Augmented_Train_images'      # Folder where the augmented traing dataset will be saved
validation_folder = 'Validation_Images880'             # Path to folder with the enhanced validation set
p_validation_folder = 'P_Validation_Images880'         # Path to folder with the original validation set
train_csv = 'train_data_classes.csv'                   # CSV with annotations for training set
validation_csv = 'validation_data_classes.csv'         # CSV for both validation sets is the same
p_validation_csv = 'validation_data_classes.csv'       # CSV for both validation sets is the same
model_file = 'MOBNETv2---RICH---Undubious---SAVING---2valsets--UNFREEZE-0-to-120---SGD-LR5e-5-MM07---D15-L002-E120-BATCH12.h5' # File name of the model to be saved (after trainING)
result_csv = 'prediction_results.csv'                 # Generated CSV to check the predicitions image-by-image                        

# Define parameteres
image_width, image_height = 240, 320    # Set to the original image size (240x320)
n_eps = 120                              # Nº of epochs
imgs_per_batch = 12                     # Batch Size




## Generate all training and testing data ---------------------------------------------------------------------------- ##

# Read training annotations
train_annotations = pd.read_csv(train_csv)                          # Reads CSV file with training data into a Dataframe
train_annotations.columns = train_annotations.columns.str.strip()   # strips whitespace - removes leading and trailing spaces
train_annotations['Fish'] = train_annotations['Fish'].astype(str)   # converts 'Fish' column to string - garantees labels are in string format

# Create augmented_train_folder if it does not exist
if not os.path.exists(augmented_train_folder):
    os.makedirs(augmented_train_folder)

# Unique classes in the dataset
unique_classes = train_annotations['Fish'].unique()  # two unique classes in the 'Fish' column of annotations file: 0 ('None') and 1 ('Fish')

# Create subdirectories for each unique class ('0' and '1') in the augmented_train_folder
for class_name in unique_classes:
    class_dir = os.path.join(augmented_train_folder, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# TensorFlow's Keras API for data augmentation -> Adjust ImageDataGenerator to use the MobileNetV2 preprocessing function
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # preprocess_input function from TensorFlow Keras library (tensorflow.keras.applications)
    rotation_range=20, 
    horizontal_flip=True, 
    vertical_flip=True, 
    shear_range=0.1, 
    fill_mode='nearest')                   
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)   # No augmentations for validation (only preprocess)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # No augmentations for validation (only preprocess)
# ImageDataGenerator with preprocessing_function=preprocess_input, the function is applied to each image as it is loaded. -> 
    # -> This process happens dynamically to ensure the raw images are transformed appropriately before being passed to the model. 

# Augment and save images in respective class subdirectories
for index, row in train_annotations.iterrows():
    img_path = os.path.join(train_folder, row['filename'])                  # Access value of 'filename' column in DataFrame train_annotations        
    img = image.load_img(img_path, target_size=(image_width, image_height)) # Image is loaded (and resized if needed)
    x = image.img_to_array(img)                                             # Transform image into NumPy array
    x = np.expand_dims(x, axis=0)                                           # Adds a batch dimension (batch for augmentation)

    class_subdir = os.path.join(augmented_train_folder, row['Fish'])        # Directory to save augmented images is determined by class

    # Determine the number of augmented images to create based of class (4 augmentations for 'None', 1 for 'Fish')
    num_augmented_images = 4 if row['Fish'] == '0' else 1

    # Save original image in augmented folder
    original_img_filename = 'original_' + row['filename']
    img.save(os.path.join(class_subdir, original_img_filename))

    # Generate augmented images
    for i in range(num_augmented_images):
        # Uses the ImageDataGenerator instance train_datagen - augmented images are saved directly in the class-specific subdirectory
        batch = train_datagen.flow(x, 
            batch_size=1, save_to_dir=class_subdir,     # batch_size for augmnetation = 1
            save_prefix='aug', 
            save_format='jpeg')
        batch.next()                                    # Retrieve next batch

# Generator that loads augmented images from the directory structure for training
train_generator = train_datagen.flow_from_directory(
    directory=augmented_train_folder,
    target_size=(image_width, image_height),
    batch_size=imgs_per_batch,
    class_mode='binary')

# Recalculate the number of training images (now with augmentation)
num_train_images = sum([len(files) for r, d, files in os.walk(augmented_train_folder)])
train_steps_per_epoch = np.ceil(num_train_images / imgs_per_batch)

# Load 1st validation annotations
validation_annotations = pd.read_csv(validation_csv)                         # Reads CSV file with validation data into a Dataframe    
validation_annotations.columns = validation_annotations.columns.str.strip()  # strips whitespace - removes leading and trailing spaces
validation_annotations['Fish'] = validation_annotations['Fish'].astype(str)  # converts 'Fish' column to string - garantees labels are in string format

# Load 2nd validation set
p_validation_annotations = pd.read_csv(p_validation_csv)
p_validation_annotations.columns = p_validation_annotations.columns.str.strip()
p_validation_annotations['Fish'] = p_validation_annotations['Fish'].astype(str)

# Data generator that loads augmented images from the directory structure for 1st validation
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_annotations, 
    directory=validation_folder, 
    x_col='filename', y_col='Fish', 
    target_size=(image_width, image_height), 
    batch_size=imgs_per_batch, 
    class_mode='binary')
validation_steps = np.ceil(validation_annotations.shape[0] / imgs_per_batch)

# Data gsenerator that loads augmented images from the directory structure for 2nd validation
p_validation_generator = val_datagen.flow_from_dataframe(dataframe=p_validation_annotations, 
    directory=p_validation_folder, 
    x_col='filename', 
    y_col='Fish', 
    target_size=(image_width, image_height), 
    batch_size=imgs_per_batch, 
    class_mode='binary')
p_validation_steps = np.ceil(p_validation_annotations.shape[0] / imgs_per_batch)

# Define a custom callback to evaluate on the 2nd validation set (1st validation set does not require callback - already supported)
class PValidationCallback(Callback):            # Create new class for callback
    def on_epoch_end(self, epoch, logs=None):   
        # Overrides on_epoch_end method from base Callback - automatically called by TensorFlow after each epoch ->
            # -> 'epoch' parameter is the index of the recently finished epoch ->
            # -> 'logs' is a dictionary with info about the current training state (such as metrics)
        logs = logs or {} 
            # ensures that logs is a dictionary
        val_loss, val_acc = self.model.evaluate(p_validation_generator, steps=p_validation_steps)   
            # p_validation_generator is the data source and p_validation_steps specifies how data batches to draw from the generator for the evaluation ->
            # evaluate method returns the loss value (val_loss) and accuracy (val_acc) on this additional validation set
        logs['p_val_loss'] = val_loss
        logs['p_val_accuracy'] = val_acc  
            # add the obtained validation loss and accuracy to the logs dictionary under the keys 'p_val_loss'
        print(f"Epoch {epoch+1}: P-Validation loss: {val_loss}, P-Validation accuracy: {val_acc}")
            # prints out a message to the console showing the loss and accuracy

# Define the base pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    # initializes MobileNetV2 as the base model (a lightweight deep neural network known for its efficiency)
    # weights='imagenet' -> Model is loaded with weights pre-trained on the ImageNet dataset (learned from a large benchmark dataset)
    # include_top=False -> Excludes top (or last fully connected) ImageNet-specific layers of the model, as we add custom layers for binary classification



## Build and train the model ---------------------------------------------------------------------------- ##

# Assuming 'base_model' is your MobileNetV2 model without the top layer
# Total number of layers in the base model
total_layers = len(base_model.layers)
print(f"Total layers: {total_layers}")

# Specify layers to unfreeze, counting from the top/end (Python's 0-indexed system)
# layers_to_unfreeze = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]  # FIRST TOP LAYER HAS INDEX 0

# Specify layers to unfreeze, counting from the top/end (Python's 0-indexed system)
layers_to_unfreeze = list(range(0, 120))

# Printing the array
print(layers_to_unfreeze)

# Freeze all layers first
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze only the specified layers
# Note: No need to adjust indices since we're directly specifying them based on your requirement
for index in layers_to_unfreeze:
    base_model.layers[-(index + 1)].trainable = True

# Build the model
model = Sequential([                # Sequential Model -> Defines a new model as a linear stack of layers
   
    base_model,                     # Base model is MobileNetV2
    
    GlobalAveragePooling2D(),       # GlobalAveragePooling2D() -> Layer that reduces each feature map to a single value ->  
                                        # -> minimizes overfitting by reducing total number of model parameters in the model

    Dropout(0.15),                   # Adjust dropout rate
                                        
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.002))  # Dense layer with L2 regularization  -> A fully connected layer with a single output unit ->
                                        # -> Sigmoid activation outputs a probability of the input image belonging to the positive class (fish)
 
    # Dense(1, activation='sigmoid')                                     
])

# Callback for early stopping - monitor 'val_accuracy' and stop training if it does not improve for 245 epochs
early_stopping = EarlyStopping(monitor='val_accuracy', patience=245, verbose=1, mode='max', restore_best_weights=True)
\
# Callback for model checkpointing - save the best model based on 'val_accuracy'
checkpoint_filepath = 'best_model_based_on_val_acc.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Reduce learning rate when training loss has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.25, patience=5, verbose=1, mode='min')

# Compile the model with a possibly lower learning rate
# model.compile(optimizer=Adam(learning_rate=0.000002), loss='binary_crossentropy', metrics=['accuracy'])
    # Optimizer -> Adam is an adaptive optimization algorithm (here set with a specific initial learning rate)
    # Loss function -> binary_crossentropy is appropriate for binary classification tasks
    # Accuracy is chosen to evaluate model performance

# Using SGD with momentum instead of Adam as an optmizer
optimizer = SGD(learning_rate=5e-5, momentum=0.7)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Include the custom callback for the second validation set in the fit method
p_validation_callback = PValidationCallback()

# Train the model with the new callbacks included
history = model.fit(train_generator, 
                    steps_per_epoch=train_steps_per_epoch, 
                    epochs=n_eps, 
                    validation_data=validation_generator, 
                    validation_steps=validation_steps, 
                    callbacks=[p_validation_callback, early_stopping, model_checkpoint])

#After training, save the final model
model.save(model_file)

# After training, save/load the best model
model.save(checkpoint_filepath)
best_model = load_model(checkpoint_filepath)
# model.save(checkpoint_filepath)



## Save Metrics ------------------------------------------------------------------------------------- ##

# Access standard validation metrics from history object
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Access custom validation metrics from history object
# These lines assume that p_val_accuracy and p_val_loss are correctly set in the logs dictionary by your PValidationCallback
p_val_accuracy = history.history['p_val_accuracy']
p_val_loss = history.history['p_val_loss']

# Compile all metrics into a DataFrame
metrics_df = pd.DataFrame({
    'Epoch': range(1, len(val_accuracy) + 1),
    'Val_Accuracy': val_accuracy,
    'P_Val_Accuracy': p_val_accuracy,
    'Val_Loss': val_loss,
    'P_Val_Loss': p_val_loss
})

# Define your desired CSV file name
metrics_csv_filename = 'training_metrics.csv'

# Save the DataFrame to CSV
metrics_df.to_csv(metrics_csv_filename, index=False)

print(f'Metrics saved to {metrics_csv_filename}')



## Load and Plot Metrics ---------------------------------------------------------------------------- ##

# Load the metrics from the CSV file
metrics_csv_filename = 'training_metrics.csv'  # Make sure this matches the filename used when saving the CSV
metrics_df = pd.read_csv(metrics_csv_filename)

# Plot validation accuracy
plt.figure(num='1', figsize=(8, 6))
plt.plot(metrics_df['Epoch'], metrics_df['Val_Accuracy'], label='Enhanced') #, marker='o')
plt.plot(metrics_df['Epoch'], metrics_df['P_Val_Accuracy'], label='Original', linestyle='--') #, marker='o')
plt.title('Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.legend()
plt.legend(loc='upper left')
plt.grid(True)

# Plot validation loss
plt.figure(num='2', figsize=(8, 6))
plt.plot(metrics_df['Epoch'], metrics_df['Val_Loss'], label='Enhanced') #, marker='o')
plt.plot(metrics_df['Epoch'], metrics_df['P_Val_Loss'], label='Original', linestyle='--') #, marker='o')
plt.title('Test Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.legend()
plt.legend(loc='upper right')
plt.grid(True)

# Plot train accuracy
plt.figure(num='3', figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy') #, marker='o')
plt.title('Train Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend()
plt.legend(loc='upper left')
plt.grid(True)

# Plot train loss
plt.figure(num='4', figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss') #, marker='o')
plt.title('Train Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend()
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()
