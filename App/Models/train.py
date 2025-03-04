import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Directorio del dataset
DATASET_PATH = "../Static/dataset-resized"

# Parámetros del modelo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Puedes aumentar si tienes tiempo

# Cargar datos y aplicar aumentación
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

MODEL_PATH = "static/models/trash_classifier.h5"

if os.path.exists(MODEL_PATH):
    print("📌 Modelo encontrado. Cargando el modelo en lugar de entrenarlo nuevamente...")
    model = keras.models.load_model(MODEL_PATH)
else:
    # Cargar MobileNetV2 preentrenado
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Congelar capas del modelo base
    base_model.trainable = False

    # Crear nuevas capas superiores
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)

    # Definir modelo
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compilar modelo
    model.compile(optimizer=keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    # Entrenar modelo
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

    # Guardar modelo entrenado
    model.save("static/models/trash_classifier.h5")
    print(" Entrenamiento finalizado. Modelo guardado en 'static/models/trash_classifier.h5'.")

print("📌 Clases detectadas en el dataset:")
print(train_generator.class_indices)  # Muestra el diccionario de clases