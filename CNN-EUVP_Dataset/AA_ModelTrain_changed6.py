import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# Define paths
train_folder = 'P_Train_images'
augmented_train_folder = 'Augmented_Train_images'  # Folder for augmented images
validation_folder = 'P_Validation_Images'
test_folder = 'P_Test_Images'
train_csv = 'train_data_classes.csv'
validation_csv = 'validation_data_classes.csv'  # Path to your validation CSV file
test_csv = 'test_data_classes.csv'  # Path to your test CSV file
model_file = 'AA_chng6_poorTrain_poorVal.h5'
result_csv = 'prediction_results.csv'

# Image dimensions
#image_width, image_height = 320, 240
image_width, image_height = 128, 128
imgs_per_batch = 48
n_eps = 47

# Read training annotations
train_annotations = pd.read_csv(train_csv)
train_annotations.columns = train_annotations.columns.str.strip()
train_annotations['Fish'] = train_annotations['Fish'].astype(str)  # Convert 'Fish' column to string

# Create augmented_train_folder if it does not exist
if not os.path.exists(augmented_train_folder):
    os.makedirs(augmented_train_folder)

# Unique classes in the dataset
unique_classes = train_annotations['Fish'].unique()

# Create subdirectories for each class
for class_name in unique_classes:
    class_dir = os.path.join(augmented_train_folder, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Enhanced ImageDataGenerator with Elastic Transform
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flipping
    shear_range=0.1,     # Added slight shearing
    preprocessing_function=lambda x: elastic_transform(x, alpha=120, sigma=12), # Elastic transform
    fill_mode='nearest')

# Read validation annotations
validation_annotations = pd.read_csv(validation_csv)
validation_annotations.columns = validation_annotations.columns.str.strip()
validation_annotations['Fish'] = validation_annotations['Fish'].astype(str)  # Convert 'Fish' column to string

# Set up the ImageDataGenerator for validation data (without augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Validation and test generators
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_annotations,
    directory=validation_folder,
    x_col='filename',
    y_col='Fish',
    target_size=(image_width, image_height),
    batch_size=imgs_per_batch,
    class_mode='binary')

# Calculate the number of steps per epoch for validation
validation_steps = np.ceil(validation_annotations.shape[0] / imgs_per_batch)


# Augment and save images in respective class subdirectories
for index, row in train_annotations.iterrows():
    img_path = os.path.join(train_folder, row['filename'])
    img = image.load_img(img_path, target_size=(image_width, image_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    class_subdir = os.path.join(augmented_train_folder, row['Fish'])

    # Determine the number of augmented images to create
    num_augmented_images = 4 if row['Fish'] == '0' else 1  # 4 for 'None' (0), 1 for 'Fish' (1)

    # Save original image in augmented folder
    original_img_filename = 'original_' + row['filename']
    img.save(os.path.join(class_subdir, original_img_filename))

    # Generate augmented images
    for i in range(num_augmented_images):
        # Save the augmented images
        batch = train_datagen.flow(x, batch_size=1, save_to_dir=class_subdir, save_prefix='aug', save_format='jpeg')
        batch.next()


# Load the combined set of images for training
train_generator = train_datagen.flow_from_directory(
    directory=augmented_train_folder,
    target_size=(image_width, image_height),
    batch_size=imgs_per_batch,
    class_mode='binary')

# Recalculate the number of training images
num_train_images = sum([len(files) for r, d, files in os.walk(augmented_train_folder)])
train_steps_per_epoch = np.ceil(num_train_images / imgs_per_batch)

# Define the checkpoint path and filename
checkpoint_filepath = 'AAA_cngh6_best.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq='epoch',  # Save the model at the end of every epoch
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)


# Build CNN model with adjusted L2 Regularization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3), kernel_regularizer=l2(0.002)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.002)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.002)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.32),
    Dense(1, activation='sigmoid')
])


# Configure ReduceLROnPlateau callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=7, 
    verbose=1, 
    mode='auto',
    min_delta=0.0002, 
    cooldown=1, 
    min_lr=8e-6
)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=n_eps,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[lr_scheduler, model_checkpoint_callback]  # Add the checkpoint callback here
)

# Compile the model - here again ???
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Save the model
model.save(model_file)




# # Path to the saved model
#model_file = 'AA_chng4.h5'
# # Load the previously trained and saved model
#model = load_model(model_file)

# # Determine the epoch with the best validation accuracy
#best_epoch = np.argmax(history.history['val_accuracy']) + 1
#print(f"Best model was saved at epoch: {best_epoch}")

# # After training, load the best model
# best_model = load_model(checkpoint_filepath)
best_model = load_model(model_file)

# Load test images and make predictions
test_annotations = pd.read_csv(test_csv)
test_annotations.columns = test_annotations.columns.str.strip()
test_annotations['Fish'] = test_annotations['Fish'].astype(str)  # Convert 'Fish' column to string

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_annotations,
    directory=test_folder,
    x_col='filename',
    y_col='Fish',
    target_size=(image_width, image_height),
    batch_size=imgs_per_batch,
    class_mode='binary',
    shuffle=False
)

# Predictions
test_steps = np.ceil(test_generator.samples / test_generator.batch_size)
predictions = best_model.predict(test_generator, steps=test_steps)
#predicted_classes = (predictions > 0.5).astype(int).flatten()

# Use the numeric 'Fish' column for actual labels
actual_labels = test_annotations['Fish'].astype(int).tolist()

# Calculate TPR, FPR, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(actual_labels, predictions.flatten())

# Calculate the AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold}")

# Apply the optimal threshold to classify images
predicted_classes = (predictions > optimal_threshold).astype(int).flatten()
# Compute the confusion matrix
cm = confusion_matrix(actual_labels, predicted_classes, labels=[1, 0])

# Create a dataframe for the results
results = pd.DataFrame({
    'Image Name': test_generator.filenames,
    'Image Actual Class': ['Fish' if label == 1 else 'None' for label in actual_labels],
    'Predicted Class': ['Fish' if pred == 1 else 'None' for pred in predicted_classes],
    'Prediction Match': [(1 if actual == predicted else 0) for actual, predicted in zip(actual_labels, predicted_classes)]
})

# Save the results to a CSV file
results.to_csv(result_csv, index=False)

# Calculate and print the model's accuracy
accuracy = np.mean([actual == predicted for actual, predicted in zip(actual_labels, predicted_classes)]) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Compute the confusion matrix
cm = confusion_matrix(actual_labels, predicted_classes, labels=[1, 0])

# Calculate F1 Score
f1 = f1_score(actual_labels, predicted_classes)
print(f"F1 Score: {f1:.2f}")


# Plot the confusion matrix
plt.figure(num='1', figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fish', 'None'], yticklabels=['Fish', 'None'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot the ROC curve
plt.figure(2)
plt.figure(num='2', figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


