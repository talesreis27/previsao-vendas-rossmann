# Previsao de Vendas XPTO - Rossmann Store Sales

Projeto de series temporais e inteligencia artificial para previsao de vendas diarias usando o dataset realista **Rossmann Store Sales**.

## Objetivo

Prever as vendas diarias da Loja 1 da Rossmann e apoiar decisoes de stock e planeamento comercial para periodos de maior procura, incluindo novembro e dezembro.

O projeto foi desenvolvido para cumprir os requisitos do caso pratico:

- usar dados historicos de vendas;
- manter pelo menos 500 registos;
- realizar limpeza e pre-processamento;
- treinar modelos de series temporais e rede neuronal;
- avaliar os modelos em conjunto de teste;
- gerar previsao futura para o periodo de fim de ano;
- documentar limitacoes e melhorias.

## Dataset

Dataset utilizado: [Rossmann Store Sales - Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)

Arquivos principais:

- `data/raw/train.csv`
- `data/raw/store.csv`
- `data/raw/test.csv`

Foi selecionada a **Loja 1**, com serie diaria completa. Diferente da primeira versao do projeto, os dias com loja fechada e vendas iguais a zero foram mantidos para preservar a frequencia diaria da serie e evitar enviesamento em modelos sazonais.

## Estrutura do Projeto

```text
Projeto_Series_Temporais/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── store.csv
│   │   └── test.csv
│   └── processed/
│       ├── processed_data.csv
│       ├── train_data.csv
│       └── test_data.csv
├── models/
│   ├── lstm_model.keras
│   └── sales_scaler.pkl
├── notebooks/
│   ├── 00_check_rossmann_dataset.py
│   ├── 01_explore_data.py
│   ├── 02_process_features.py
│   ├── 03_train_test_split.py
│   ├── 04_train_ARIMA.py
│   ├── 05_train_SARIMA.py
│   ├── 06_train_LSTM.py
│   └── 07_predict_future_LSTM.py
├── outputs/
│   ├── arima_results.csv
│   ├── sarima_results.csv
│   ├── lstm_results.csv
│   ├── lstm_future_forecast.csv
│   ├── lstm_forecast_nov_dec_2015.csv
│   └── graficos/
├── src/
│   └── requirements.txt
└── README.md
```

## Metodologia

1. Verificacao do dataset Rossmann.
2. Analise exploratoria da Loja 1.
3. Manutencao da serie diaria completa, incluindo dias fechados.
4. Criacao de variaveis de calendario, lags e medias moveis.
5. Divisao cronologica treino/teste, sem embaralhamento.
6. Treino dos modelos ARIMA, SARIMA e LSTM.
7. Avaliacao com MAE, RMSE e MAPE seguro.
8. Previsao futura ate dezembro de 2015.

## Correcao de Data Leakage

Na LSTM, a normalizacao foi corrigida para evitar data leakage:

- o `MinMaxScaler` e ajustado apenas no periodo de treino;
- a serie completa e transformada usando esse scaler;
- o scaler e salvo em `models/sales_scaler.pkl`;
- o mesmo scaler e reutilizado no script de previsao futura.

## Resultados

Metricas no conjunto de teste:

| Modelo | MAE | RMSE | MAPE seguro |
|---|---:|---:|---:|
| ARIMA(1,1,1) | 1306.89 | 1924.73 | 14.62% |
| SARIMA(1,1,1)(1,1,1,7) | 705.99 | 1112.04 | 15.36% |
| LSTM | 1004.50 | 1445.16 | 14.68% |

O **SARIMA** apresentou o melhor desempenho geral em MAE e RMSE, alem de lidar melhor com a sazonalidade semanal e os dias de loja fechada.

## Previsao Futura

A previsao futura foi gerada de `2015-08-01` a `2015-12-31` usando a LSTM corrigida.

Para novembro e dezembro de 2015:

| Mes | Total previsto |
|---|---:|
| Novembro | 101427.52 |
| Dezembro | 106167.42 |

## Como Executar

Instalar dependencias:

```bash
pip install -r src/requirements.txt
```

Executar os scripts em ordem:

```bash
python notebooks/00_check_rossmann_dataset.py
python notebooks/01_explore_data.py
python notebooks/02_process_features.py
python notebooks/03_train_test_split.py
python notebooks/04_train_ARIMA.py
python notebooks/05_train_SARIMA.py
python notebooks/06_train_LSTM.py
python notebooks/07_predict_future_LSTM.py
```

## Limitacoes

- O estudo foi focado apenas na Loja 1.
- A previsao futura da LSTM e recursiva e pode acumular erro.
- A LSTM usa a sequencia historica de vendas, mas nao recebe informacoes futuras de promocoes ou feriados.
- Variaveis do `store.csv`, como tipo de loja e distancia da concorrencia, seriam mais uteis em um modelo multiloja.

## Melhorias Futuras

- Criar um modelo multiloja.
- Usar SARIMAX com variaveis externas.
- Testar XGBoost, Random Forest ou LSTM multivariada.
- Usar validacao temporal com multiplas janelas.
- Incorporar calendario futuro de promocoes, feriados e dias de abertura.

