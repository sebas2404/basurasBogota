import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image

# Obtener el directorio base del archivo classifier.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo entrenado
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "garbage_classifier.h5")

# Verificar si el modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo no encontrado en: {MODEL_PATH}")

print(f"✅ Modelo encontrado en: {MODEL_PATH}")

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Diccionario de clases
class_labels = {
    0: "BATTERY",
    1: "BIOLOGICAL",
    2: "CARDBOARD",
    3: "CLOTHES",
    4: "GLASS",
    5: "METAL",
    6: "PAPER",
    7: "PLASTIC",
    8: "SHOES",
    9: "TRASH"
}

def classify_image(img_path):
    """
    Carga una imagen, la preprocesa y la clasifica con el modelo.
    """
    img_path = os.path.abspath(img_path)  # Asegurar ruta absoluta

    # Leer la imagen con OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("❌ Error al cargar la imagen. Verifica la ruta.")

    # Convertir BGR a RGB y redimensionar
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Convertir a array, normalizar y expandir dimensiones
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    prediction = model.predict(img_array)[0]  # Obtener solo el primer resultado
    predicted_class = np.argmax(prediction)
    class_name = class_labels[predicted_class]
    confidence = prediction[predicted_class] * 100  # Convertir a %

    return class_name, confidence, prediction

def process_image(img_path):
    """
    Procesa la imagen dada, la clasifica y retorna los resultados.
    """
    if not os.path.exists(img_path):
        print("❌ Error: La imagen no existe.")
        return None

    class_name, confidence, prediction = classify_image(img_path)
    
    print(f"\n📌 La imagen es clasificada como: **{class_name}** ({confidence:.2f}% de confianza)")

    # Mostrar todas las probabilidades
    print("\n📊 Probabilidades por clase:")
    probabilities = {}
    for idx, prob in enumerate(prediction):
        prob_percentage = prob * 100
        probabilities[class_labels[idx]] = prob_percentage
        print(f"   - {class_labels[idx]}: {prob_percentage:.2f}%")

    return class_name, confidence, probabilities

if __name__ == "__main__":
    img_path = input("Ingrese la ruta de la imagen: ").strip()
    process_image(img_path)