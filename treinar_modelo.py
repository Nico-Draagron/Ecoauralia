import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Directory of images
img_folder = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\spectrogramas"

# Data Augmentation
aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

# Training and validation data
train_gen = aug.flow_from_directory(
    img_folder,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

valid_gen = aug.flow_from_directory(
    img_folder,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Balanced weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(zip(np.unique(train_gen.classes), weights))

print("Adjusted weights:", class_weights)

# Load and freeze MobileNet
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 3)))
for layer in mobilenet.layers:
    layer.trainable = False

# Custom head
x = mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
prediction = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=mobilenet.input, outputs=prediction)

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_gen,
    epochs=40,
    validation_data=valid_gen,
    class_weight=class_weights
)

# Save
model.save("balanced_mobilenet_model.h5")
print("🎉 Done! Your MobileNetV2 model is trained and saved successfully!")
