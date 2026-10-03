import os
import re
import cv2
import numpy as np
import face_recognition
import easyocr
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

app = FastAPI()

# Cargar los modelos al iniciar el servidor en la nube
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['es'])

# IMPORTANTE: Puedes cambiar estas matrículas cuando quieras editando este archivo
MATRICULAS_FAMILIA = ["1234ABC", "5678XYZ"]

def cargar_rostros_carpeta(ruta_carpeta):
    encodings_lista = []
    if os.path.exists(ruta_carpeta):
        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith(('.jpg', '.jpeg', '.png')):
                ruta_foto = os.path.join(ruta_carpeta, archivo)
                try:
                    img = face_recognition.load_image_file(ruta_foto)
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings) > 0:
                        encodings_lista.append(encodings)
                except Exception:
                    pass
    return encodings_lista

# Cargar imágenes estáticas de referencia de las carpetas internas
rostros_familia = cargar_rostros_carpeta("familia")
rostros_conocidos = cargar_rostros_carpeta("conocidos")

def limpiar_texto_matricula(texto):
    return re.sub(r'[^A-Z0-9]', '', texto.upper())

@app.post("/analizar")
async def analizar_seguridad(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"evento": "NADA", "motivo": "Imagen ilegible"}

    # 1. DETECCIÓN CON YOLO (Filtrar animales automáticamente)
    resultados = model(img, verbose=False)
    hay_humanos = False
    hay_coches = False
    coordenadas_coches = []

    for box in resultados.boxes:
        clase = model.names[int(box.cls)]
        if clase == 'person':
            hay_humanos = True
        elif clase == 'car':
            hay_coches = True
            x1, y1, x2, y2 = map(int, box.xyxy)
            coordenadas_coches.append((x1, y1, x2, y2))

    if not hay_humanos and not hay_coches:
        return {"evento": "NADA", "motivo": "Solo animales o entorno vacío"}

    # 2. ANÁLISIS DE MATRÍCULAS (Si no se lee con éxito, se descarta)
    hay_coche_extrano = False
    hay_coche_familia = False

    if hay_coches:
        for (x1, y1, x2, y2) in coordenadas_coches:
            recorte_coche = img[y1:y2, x1:x2]
            resultados_ocr = reader.readtext(recorte_coche)
            if len(resultados_ocr) == 0:
                continue
                
            matricula_detectada = False
            for (_, texto, _) in resultados_ocr:
                texto_limpio = limpiar_texto_matricula(texto)
                for mat_fam in MATRICULAS_FAMILIA:
                    if mat_fam in texto_limpio or texto_limpio in mat_fam:
                        if len(texto_limpio) >= 4:
                            hay_coche_familia = True
                            matricula_detectada = True
                            break
            if not matricula_detectada:
                hay_coche_extrano = True

    # 3. ANÁLISIS DE ROSTROS
    hay_familia_presente = False
    hay_conocido_presente = False
    hay_extrano_presente = False

    if hay_humanos:
        rgb_camara = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rostros_en_vivo = face_recognition.face_encodings(rgb_camara)

        if len(rostros_en_vivo) == 0:
            return {"evento": "NADA", "motivo": "Humano detectado de espaldas o rostro oculto"}

        for rostro_detectado in rostros_en_vivo:
            if len(rostros_familia) > 0:
                comp_fam = face_recognition.compare_faces(rostros_familia, rostro_detectado, tolerance=0.55)
                if any(comp_fam):
                    hay_familia_presente = True
                    continue

            if len(rostros_conocidos) > 0:
                comp_con = face_recognition.compare_faces(rostros_conocidos, rostro_detectado, tolerance=0.55)
                if any(comp_con):
                    hay_conocido_presente = True
                    continue

            hay_extrano_presente = True

    # 4. APLICACIÓN DE TUS REGLAS DE SEGURIDAD STRICTAS
    if hay_familia_presente or hay_coche_familia:
        return {"evento": "NADA", "motivo": "Familia o vehículo familiar presente."}

    if hay_conocido_presente and not hay_extrano_presente and not hay_coche_extrano:
        return {"evento": "AVISO", "motivo": "Conocido detectado a pie."}

    if hay_extrano_presente or hay_coche_extrano:
        return {"evento": "AVISO_Y_ALARMA", "motivo": "¡Intrusión! Persona o coche no autorizado."}

    return {"evento": "NADA", "motivo": "Escena segura."}
  
