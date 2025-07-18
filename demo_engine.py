import joblib
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

try:
    # 1. Cargar el vectorizador y las etiquetas
    vectorizer = joblib.load('vectorizer.joblib')
    with open('labels_y_titulo.pkl', 'rb') as f:
        titulares = list(pickle.load(f))

    # 2. Reconstruir la matriz desde el .npz
    loader = np.load('features_X.npz')
    X_full = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    
    # 3. Ajuste de dimensiones (3000 columnas del vectorizador)
    X_dataset = X_full[:, :3000]

    # 4. Texto de entrada (Simulando un titular que sabemos que existe)
    texto_entrada = "Renesas sets new MCU performance bar with RA8P1 devices AI acceleration"

    # 5. Transformar y comparar
    vector_nuevo = vectorizer.transform([texto_entrada])
    similitudes = cosine_similarity(vector_nuevo, X_dataset)
    indice_max = np.argmax(similitudes)
    confianza = similitudes[0][indice_max]

    print("\n" + "="*60)
    print(f"🔍 INPUT: {texto_entrada}")
    print(f"🎯 MATCH ENCONTRADO: {titulares[indice_max]}")
    print(f"📊 CONFIANZA (Similitud de Coseno): {confianza:.4f}")
    print("="*60 + "\n")

except Exception as e:
    print(f"❌ Error: {e}")
