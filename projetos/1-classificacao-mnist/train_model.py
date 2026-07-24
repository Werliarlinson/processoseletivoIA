import os
# Desabilita completamente a busca por GPU, evitando travamentos no contêiner
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Oculta os logs e avisos do TensorFlow em C++ (exibe apenas erros críticos)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

def main():
    print("[1/7] Carregando o dataset MNIST...")
    # O dataset já vem dividido entre dados de treino e dados de teste invisíveis
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print("[2/7] Normalizando e ajustando as dimensões das imagens...")
    # Convertendo pixels de 0-255 para 0.0-1.0 (melhora a estabilidade numérica)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Ajustando de (28, 28) para (28, 28, 1) para informar que há apenas 1 canal de cor (escala de cinza)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print("[4/7] Construindo a arquitetura da CNN...")
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        
        # Bloco Convolucional 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco Convolucional 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco Convolucional 3
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Preparação para a Camada de Saída
        layers.Flatten(),
        layers.Dropout(0.5),                     # Regularização para evitar overfitting
        layers.Dense(10, activation="softmax")   # 10 neurônios de saída (dígitos de 0 a 9)
    ])

    # Compilando o modelo
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    # Configurando o Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,                 # Tolerância de 3 épocas sem melhora
        restore_best_weights=True   # Ao finalizar, reverte para os pesos da melhor época
    )

    print("[3/7 e 5/7] Iniciando o treinamento com validação e Early Stopping...")
    # O validation_split=0.2 tira 20% do x_train para ser usado como validação (Requisito 3)
    model.fit(
        x_train, y_train,
        epochs=15, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\n[6/7] Avaliando a acurácia final...")
    # Avaliamos o modelo no conjunto x_test que ficou totalmente isolado durante o treino
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("-" * 50)
    print(f"✅ Acurácia final no conjunto de validação/teste: {test_acc:.4f} ({(test_acc * 100):.2f}%)")
    print("-" * 50)

    print("[7/7] Salvando o modelo...")
    model.save("model.h5")
    print("✅ Processo concluído! Modelo salvo como 'model.h5'")

# Ponto de entrada do script
if __name__ == "__main__":
    main()