# Projeto ISIC 2024 Challenge

## Visão Geral
Este projeto foi desenvolvido para processar e analisar dados para o desafio ISIC 2024. Ele inclui scripts para pré-processamento de dados, ajuste de hiperparâmetros, seleção de características e execução de modelos de aprendizado de máquina, como MLP, GBDT e Naive Bayes.

## Instruções de Configuração

### 1. Criar o Ambiente
Para configurar o ambiente, utilize o arquivo `environment.yml` fornecido no projeto. Execute o seguinte comando:

```bash
conda create -n isic2024 python=3.12.7
conda activate isic2024
```

### 2. Modificar o Arquivo de Configuração
Edite o arquivo `config.py` para definir os parâmetros apropriados para o seu experimento. Configure o RAW_METADATA_PATH com o caminho do [train-metadata.csv](https://www.kaggle.com/competitions/isic-2024-challenge/data?select=train-metadata.csv). Além disto, escolha um caminho para os dados preprocessados em ONE_HOT_ENCODED_PATH.

### 3. Executar o Script de Pré-Processamento
Para pré-processar os dados, execute o script `preprocess.py`:

```bash
python preprocess.py
```

### 4. (Opcional) Ajustar Hiperparâmetros
Se desejar ajustar os hiperparâmetros do GBDT, execute o script `hyperparameters.py`:

```bash
python hyperparameters.py
```

### 5. (Opcional) Realizar Seleção de Características
Para realizar a seleção de características para o modelo Naive Bayes, execute o script `nb_feature_selection.py`:

```bash
python nb_feature_selection.py
```

### 6. Executar o Modelo MLP
Para treinar e avaliar o modelo MLP, execute o script `mlp.py`:

```bash
python mlp.py
```

### 7. Executar o Modelo GBDT
Para treinar e avaliar o modelo GBDT, execute o script `gbdt.py`:

```bash
python gbdt.py
```

### 8. Executar o Modelo Naive Bayes
Para treinar e avaliar o modelo Naive Bayes, execute o script `nb.py`:

```bash
python nb.py
```

### 9. Agregar os resultados e realizar testes estatísticos
Para agregar os resultados e realizar testes estatísticos, você precisará alterar o `config.py` para incluir os caminhos corretos para os arquivos de resultados em EXPERIMENT_DIRS_BY_METHOD. Em seguida, execute o script `aggregate_metrics.py`:

```bash
python aggregate_metrics.py
```


## Resultados
Os resultados dos experimentos, incluindo curvas de precisão-recall e ROC, são salvos no diretório `results/`, organizados por modelo e timestamp.

## Notas
- Certifique-se de que os dados necessários estejam disponíveis e devidamente formatados antes de executar os scripts.
- Consulte os arquivos de script individuais para obter informações adicionais sobre parâmetros e opções.
