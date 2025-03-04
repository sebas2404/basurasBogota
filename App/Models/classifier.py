import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Cargar el modelo entrenado


# Obtener el directorio base del proyecto
# Obtener el directorio base del archivo classifier.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta corregida al modelo
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "trash_classifier.h5")

# Verificar si el archivo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo no encontrado en: {MODEL_PATH}")

print(f"✅ Modelo encontrado en: {MODEL_PATH}")


model = tf.keras.models.load_model(MODEL_PATH)

# Diccionario de clases
class_labels = {
    0: "CARDBOARD",
    1: "GLASS",
    2: "METAL",
    3: "PAPER",
    4: "PLASTIC",
    5: "TRASH"
}

def classify_image(img_path):
    """
    Carga una imagen, la preprocesa y la clasifica con el modelo.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Agregar batch dimension
    
    prediction = model.predict(img_array)  # Hacer predicción
    predicted_class = np.argmax(prediction)  # Obtener la clase con mayor probabilidad
    class_name = class_labels[predicted_class]
    
    return class_name, prediction[0][predicted_class]

if __name__ == "__main__":
    img_path = input("Ingrese la ruta de la imagen: ")
    if os.path.exists(img_path):
        class_name, confidence = classify_image(img_path)
        print(f"La imagen es clasificada como: {class_name} ({confidence:.2f} de confianza)")
    else:
        print("❌ Error: La imagen no existe.")
