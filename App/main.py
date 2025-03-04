import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Models.classifier import classify_image
from funcionalidad.func1 import process_image as proces_model_1
from func2.funcio2 import process_image as proces_model_2
from PIL import Image

def main():
    # Título de la página
    st.title("Gestión de Residuos Inteligente")
    
    # Selector para elegir la funcionalidad
    opcion = st.radio("Selecciona una opción:", ["Clasificador de Residuos", "Clasificador Avanzado"], horizontal=True)
    
    # Contenedor principal donde se mostrará la funcionalidad elegida
    st.divider()
    if opcion == "Clasificador de Residuos":
        mostrar_clasificador()
    elif opcion == "Clasificador Avanzado":
        mostrar_avanzado()

def mostrar_clasificador():
    st.header("Clasificador de Residuos")
    
    uploaded_file = st.file_uploader("Sube una imagen de residuo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        TEMP_DIR = "static/temp"

        # Crear la carpeta si no existe
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        
        # Guardar la imagen temporalmente
        temp_path = os.path.join("static", "temp", uploaded_file.name)
        image.save(temp_path)
        
        # Clasificar imagen
        class_name, confidence = classify_image(temp_path)
        
        # Mostrar resultado
        st.subheader("Resultado de la Clasificación")
        st.write(f"**Categoría:** {class_name}")
        st.write(f"**Confianza:** {confidence:.2f}")

        # Borrar la imagen temporal después de clasificarla
        os.remove(temp_path)

def mostrar_avanzado():
    st.header("Clasificador Avanzado de Residuos")
    
    uploaded_file = st.file_uploader("Sube una imagen de residuo para clasificación avanzada", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        TEMP_DIR = "static/temp"

        # Crear la carpeta si no existe
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        
        # Guardar la imagen temporalmente
        temp_path = os.path.join("static", "temp", uploaded_file.name)
        image.save(temp_path)
        
        # Clasificar imagen con ambos modelos
        class_name_1, confidence_1, probabilities_1 = proces_model_1(temp_path)
        class_name_2, confidence_2, probabilities_2 = proces_model_2(temp_path)
        
        # Mostrar resultados en dos divs
        st.subheader("Resultados de Clasificación")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Modelo 1: Clasificación General")
            st.write(f"**Categoría:** {class_name_1}")
            st.write(f"**Confianza:** {confidence_1:.2f}%")
            st.write("#### Probabilidades:")
            for category, prob in probabilities_1.items():
                st.write(f"{category}: {prob:.2f}%")
        
        with col2:
            st.write("### Modelo 2: Clasificación Específica")
            st.write(f"**Categoría:** {class_name_2}")
            st.write(f"**Confianza:** {confidence_2:.2f}%")
            st.write("#### Probabilidades:")
            for category, prob in probabilities_2.items():
                st.write(f"{category}: {prob:.2f}%")

        # Borrar la imagen temporal después de clasificarla
        os.remove(temp_path)

if __name__ == "__main__":
    main()
