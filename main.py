import cv2
import numpy as np
import time
import os

def alinear_canales(canal_ref, canal_mover):
    # Extraemos solo el centro de la imagen para evitar que los bordes dañados de las placas de cristal arruinen la correlación,
    h, w = canal_ref.shape
    cx, cy = w // 2, h // 2
    d = min(w, h) // 4  # Usamos una ventana central
    
    ref_centro = np.float32(canal_ref[cy-d:cy+d, cx-d:cx+d])
    mover_centro = np.float32(canal_mover[cy-d:cy+d, cx-d:cx+d])

    # Correlación de fase (Fourier)
    (dx, dy), _ = cv2.phaseCorrelate(mover_centro, ref_centro)

    # Otras correlaciones (Comentadas según 2.3)
    # Ahora mismo no van correctamente, hay que modificar el return

    # 1. Correlación Normalizada (Normalized Cross-Correlation)
    # res_ncc = cv2.matchTemplate(mover_centro, ref_centro, cv2.TM_CCORR_NORMED)
    # _, _, _, max_loc = cv2.minMaxLoc(res_ncc)
    # dx, dy = max_loc[0] - d, max_loc[1] - d
    
    # 2. Suma de Diferencias al Cuadrado (SSD)
    # res_ssd = cv2.matchTemplate(mover_centro, ref_centro, cv2.TM_SQDIFF)
    # min_val, _, min_loc, _ = cv2.minMaxLoc(res_ssd)
    # dx, dy = min_loc[0] - d, min_loc[1] - d

    return int(round(dx)), int(round(dy))
    # return dx, dy

def mejoras_extra(imagen_bgr): # Recorte de marcos, eliminación de defectos y corrección fotométrica (contraste).
    # 1. Eliminación de bordes y marcos (Recortamos un 6% de los márgenes)
    alto, ancho = imagen_bgr.shape[:2]
    margen_y = int(alto * 0.06)
    margen_x = int(ancho * 0.06)
    img_recortada = imagen_bgr[margen_y:alto-margen_y, margen_x:ancho-margen_x]

    # 2. Eliminación de defectos (Filtro de mediana para quitar polvo/ruido) Kernel 3 es suficiente
    img_limpia = cv2.medianBlur(img_recortada, 3)

    # 3. Corrección fotométrica (Ecualización adaptativa CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Separamos los canales B, G, R para aplicar el contraste a cada uno
    b, g, r = cv2.split(img_limpia)

    # Volvemos a unir los canales y devolvemos
    return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])


def procesar_imagen(ruta_imagen):
    # Tasca 1 - Escollir el dataset / Lectura de la imagen
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    print(f"\nTrabajando con: {ruta_imagen}")

    t0 = time.time() # Calculo del tiempo (Se pide)

    # Tasca 3 - Implementar la solucio.

    # 3.1. Recorte de subimágenes
    # Vamos a divir la altura total entre 3 (B, G, R).
    alto_canal = img.shape[0] // 3
    B , G, R = img[0:alto_canal], img[alto_canal:2*alto_canal], img[2*alto_canal:3*alto_canal]

    # 3.2. Puesta en correspondencia (Registrado)
    # Calculamos desplazamientos usando B como referencia
    dx_g, dy_g = alinear_canales(B, G)
    dx_r, dy_r = alinear_canales(B, R)

    # Creamos las matrices de traslación afín
    M_g = np.float32([[1, 0, dx_g], [0, 1, dy_g]])
    M_r = np.float32([[1, 0, dx_r], [0, 1, dy_r]])

    # Aplicamos la traslación a los canales G y R
    G_alineado = cv2.warpAffine(G, M_g, (G.shape[1], G.shape[0]))
    R_alineado = cv2.warpAffine(R, M_r, (R.shape[1], R.shape[0]))

    # 3.3. Combinamos en una imagen a color (BGR por defecto)
    imagen_color = cv2.merge([B, G_alineado, R_alineado])

    t1 = time.time() # Tiempo final
    t_total = t1 - t0

    imagen_color = mejoras_extra(imagen_color) # Mejoras extra

    print(f"Tiempo final (recorte y registrado): {t_total:.4f}s")

    # Guardado de la imagen (fuera del cronómetro)
    nombre_base = os.path.splitext(ruta_imagen)[0]
    ruta_salida = f"{nombre_base}color.jpg"
    cv2.imwrite(ruta_salida, imagen_color)
    print(f"Resultado guardado como: {ruta_salida}")

    # Mostrar la imagen en pantalla (redimensionada)
    alto_disp, ancho_disp = imagen_color.shape[:2]
    escala = 800 / alto_disp # Redimension
    dim_disp = (int(ancho_disp * escala), 800)
    imagen_disp = cv2.resize(imagen_color, dim_disp, interpolation=cv2.INTER_AREA)
    
    cv2.imshow(f"Resultado - {ruta_imagen}", imagen_disp)
    cv2.waitKey(0) # Presiona cualquier tecla para cerrar la ventana
    cv2.destroyAllWindows()

# Llamada a funciones.
if __name__ == "__main__":
    lista_imagenes = ['255.jpg', '328.jpg', '499.jpg', '1099.jpg', '1246.jpg', '1822.jpg']
    for img_path in lista_imagenes:
        if os.path.exists(img_path):
            procesar_imagen(img_path)
        else:
            print(f"No se encontró la imagen: {img_path}")

