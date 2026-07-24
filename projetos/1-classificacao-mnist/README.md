# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Werliarlinson de Lima Sá Teles**

#### 1️⃣ Resumo da Arquitetura do Modelo
O modelo foi desenvolvido baseado em uma arquitetura CNN simples e eficiente. Foram utilizados 3 blocos convolucionais encadeados, cada um composto por camadas `Conv2D`, `BatchNormalization` e `MaxPooling2D`. Para mitigar o overfitting, aplicou-se uma camada de `Dropout` (0.5) antecedendo a saída densa (Softmax). A validação utilizou um split de 20% do conjunto de treino original, e o treinamento contou com um callback de `EarlyStopping` (patience=3) monitorando a função de perda de validação.

### 2️⃣ Bibliotecas Utilizadas
* TensorFlow / Keras (versão 2.12.0)
* NumPy (versão 2.4.6)
* OS (biblioteca nativa do Python para lidar com variáveis de ambiente)

### 3️⃣ Técnica de Otimização do Modelo
A otimização foi realizada convertendo o modelo original (.h5) para o formato TensorFlow Lite (.tflite) através do `tf.lite.TFLiteConverter`. Durante a conversão, aplicou-se a técnica de quantização padrão da biblioteca (`tf.lite.Optimize.DEFAULT`), focando em reduzir a precisão dos pesos para minimizar o tamanho final do arquivo de arquitetura, focando na aplicação em Edge AI.

### 4️⃣ Resultados Obtidos
* **Acurácia de Validação Final:** 98.85%
* **Tamanho do arquivo original (`model.h5`): 1.178 kb**
* **Tamanho do arquivo otimizado (`model.tflite`): 104 kb**

### 5️⃣ Comentários Adicionais (Opcional)
Durante o desenvolvimento no ambiente isolado (Dev Container), foi necessário desabilitar a varredura da biblioteca por instâncias de aceleração gráfica (GPU) via variáveis de ambiente para garantir que a execução ocorresse de forma fluida exclusivamente na CPU. O framework de Early Stopping atuou de maneira eficiente e abortou o treinamento na 9ª época.

### 6️⃣ Exemplo de Inferência
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Como observado na saída acima, o modelo quantizado manteve sua integridade e conseguiu inferir corretamente que a amostra 1 se tratava do dígito 7, sem apresentar confusão com outras classes.