import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

def main():
    print("[1/3] Carregando o modelo treinado (model.h5)...")
    model = tf.keras.models.load_model("model.h5")

    print("[2/3] Configurando o conversor para TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Aplicando a otimização exigida pelo desafio (Dynamic Range Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("[3/3] Convertendo e salvando o modelo otimizado...")
    tflite_model = converter.convert()

    # Salvando o modelo convertido em disco no formato binário ("wb")
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
        
    print("✅ Processo concluído! Modelo otimizado salvo como 'model.tflite'")

if __name__ == "__main__":
    main()