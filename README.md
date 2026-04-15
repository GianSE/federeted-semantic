# 🛰️ Comunicação Semântica Federada — Testbed de Pesquisa

> Protótipo de pesquisa que demonstra como **representações latentes** (VAE/AE)
> reduzem o volume de dados transmitidos preservando informação semântica — provando
> a hipótese central do projeto em MNIST, Fashion-MNIST e CIFAR-10.

[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)

---

## 📋 Hipótese de Pesquisa

> **"Representações latentes produzidas por um Autoencoder Variacional (VAE) podem
> reduzir significativamente a largura de banda de transmissão, preservando
> informação semântica suficiente para reconstrução precisa no receptor."**

### Evidências mensuráveis

| Métrica | Descrição | Direção |
|---|---|---|
| **MSE** | Erro quadrático médio pixel a pixel | ↓ melhor |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | ↑ melhor |
| **SSIM** | Similaridade estrutural (Wang et al., 2004) | ↑ melhor (máx 1.0) |
| **Razão de compressão** | bytes_originais / bytes_latentes | ↑ melhor |
| **Redução de banda** | (1 − 1/razão) × 100% | ↑ melhor |

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER                                                        │
│  React 18 + Vite + Tailwind CSS                                 │
│  Porta 5180 (via Nginx)                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP /api/*
┌──────────────────────────▼──────────────────────────────────────┐
│  BACKEND  (API Gateway)                                         │
│  Fastify (Node.js) — Porta 3000                                 │
│  Proxy transparente + CORS + SSE streaming                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│  ML SERVICE                                                     │
│  FastAPI + PyTorch 2.5 — Porta 8000 (interna)                   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   VAE / AE  │  │ Orchestrator │  │   Benchmark Engine     │ │
│  │   Models    │  │ (FedAvg sim) │  │ (multi-dataset eval)   │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              ↕ volumes Docker
┌─────────────────────────────────────────────────────────────────┐
│  volumes/ml_data/    — datasets, .pth weights, logs             │
│  volumes/resultados/ — experiment JSONs, figures, CSV/TeX       │
└─────────────────────────────────────────────────────────────────┘
```

### Modelos implementados

| ID | Nome | Encoder | Decoder | Loss |
|---|---|---|---|---|
| `cnn_vae` | CNN Variational AE | Conv→Pool→Conv→Pool→FC→μ,σ (32-d) | FC→Reshape→ConvTranspose×2→Sigmoid | MSE + β·KL |
| `cnn_ae`  | CNN Autoencoder    | Conv→Pool→Conv→Pool→FC (32-d)     | FC→Reshape→ConvTranspose×2→Sigmoid | MSE |

**Razão de compressão teórica (latente Float32):**

| Dataset | Original (float32) | Latente (32-d float32) | Razão |
|---|---|---|---|
| MNIST / Fashion-MNIST | 3136 B (28×28×1) | 128 B | **24.5×** |
| CIFAR-10              | 12288 B (32×32×3) | 128 B | **96×** |

Com quantização Int8 o latente fica 32 B + 4 B (escala) = 36 B → razão ainda maior.

---

## 🚀 Como Executar

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando

### 1. Subir os serviços

```bash
cd "federeted-semantic - MNIST"

# Construir e iniciar (primeira vez demora ~5 min para baixar datasets)
docker compose up --build -d

# Acompanhar logs em tempo real
docker compose logs -f ml-service
```

Acesse o dashboard em: **http://localhost:5180**

### 2. Treinar os modelos (necessário para métricas reais)

O painel de treinamento usa uma **simulação de demonstração** (rápida). Para métricas
semânticas reais no painel `/semantic` e `/benchmark`, treine os modelos:

```bash
# Treinar todos os datasets e modelos padrão (recomendado)
docker compose exec ml-service python -m app.train_local

# Treinar apenas um dataset/modelo específico
docker compose exec ml-service python -m app.train_local \
    --dataset fashion --model cnn_vae --epochs 10

# Opções disponíveis
docker compose exec ml-service python -m app.train_local --help
```

Os pesos são salvos em `volumes/ml_data/` e ficam persistidos entre reinicializações.

### 3. Comandos úteis

```bash
# Ver status dos containers
docker compose ps

# Parar tudo
docker compose down

# Reconstruir após alterações no código
docker compose up --build -d

# Remover volumes (reset completo — apaga datasets baixados e pesos!)
docker compose down -v
```

---

## 📱 Páginas do Dashboard

### ⚙ Treinamento Federado (`/`)

Controle interativo da simulação FedAvg:
- Configurar dataset, modelo, número de clientes (1–10)
- Topologia IID / Não-IID
- Simulador AWGN (ruído de canal com SNR configurável)
- Injeção de ruído: White noise, packet loss, client drift
- Terminal em tempo real por nodo (server + client-N)
- Pausa/Resume/Stop do orquestrador

> **Nota:** O treinamento no dashboard é uma simulação de demonstração. Execute
> `train_local.py` para pesos reais.

### 📊 Relatórios de Resultados (`/results`)

Navegue pelo repositório de experimentos finalizados:
- Curvas de convergência (Loss e Acurácia por round)
- Imagem de reconstrução semântica real (quando pesos disponíveis)
- Download de resultados em CSV e LaTeX
- Métricas: dataset, modelo, topologia, AWGN, acurácia final, loss final

### 📡 Comunicação Semântica (`/semantic`)

Demonstração interativa em duas etapas:

**1. Pipeline de compressão:**
- Selecione dataset, modelo e nível de quantização (4/8/16/32 bits)
- Veja: imagem original → "transmissão" → reconstrução
- Métricas: MSE, PSNR, SSIM, bytes transmitidos, razão de compressão

**2. Recuperação de erros de canal:**
- Simule perda de pacotes mascarando partes da imagem
- Tipos: metade inferior, metade direita, pixels aleatórios
- Avalie a reconstrução pelo modelo com MSE, PSNR, SSIM

### 📉 Análise Trade-off (`/tradeoff`)

Varredura Monte Carlo: SNR × quantização:
- Selecione dataset e modelo
- Gráfico: PSNR vs SNR para 4 níveis de quantização (Int4, Int8, Int16, Float32)
- Gráfico de barras: tamanho real do payload por nível
- Demonstra que Int8 mantém qualidade semelhante ao Float32 com 75% menos bytes

### 🔬 Benchmark Multi-Dataset (`/benchmark`)

**Página principal de evidência científica:**
- Executa avaliação estruturada em MNIST, Fashion-MNIST e CIFAR-10
- Tabela: Dataset | Modelo | MSE | PSNR | SSIM | Razão | Redução de Banda
- Gráfico de barras: razão de compressão por combinação
- Gráfico de barras: PSNR + SSIM comparativos
- Radar chart: VAE vs AE multidimensional
- Tabela de escalabilidade: largura de banda para 1, 5, 10, 50, 100 dispositivos
- Semente fixa (42) para reprodutibilidade

---

## 🧪 Reprodutibilidade

Todos os experimentos usam `seed=42` por padrão:

```python
# train_local.py — seeds globais
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

Para reproduzir exatamente os resultados:

```bash
docker compose exec ml-service python -m app.train_local \
    --dataset fashion --model cnn_vae --epochs 10 --seed 42
```

---

## 🔧 Configurações e Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `ML_SERVICE_URL` | `http://ml-service:8000` | URL interna do serviço Python |
| `PORT` (backend) | `3000` | Porta do Fastify |
| `DATA_ROOT` | `/ml-data` | Raiz dos dados persistidos |
| `DATASETS_DIR` | `/ml-data/datasets` | Diretório dos datasets |
| `LOGS_DIR` | `/ml-data/logs` | Logs de treinamento |
| `RESULTADOS_ROOT` | `/resultados` | Experimentos finalizados |

Copie `.env.example` para `.env` e ajuste conforme necessário.

---

## 📂 Estrutura do Projeto

```
federeted-semantic - MNIST/
├── docker-compose.yml
├── .env.example
│
├── frontend/                        # React + Vite
│   └── src/
│       ├── App.jsx                  # Roteamento + sidebar
│       └── pages/
│           ├── TrainingDashboard/   # Simulação FedAvg
│           ├── Results/             # Histórico de experimentos
│           ├── SemanticComms/       # Pipeline de compressão
│           ├── Tradeoff/            # Análise SNR × bits
│           └── Benchmark/           # ★ Evidência científica
│
├── backend/                         # Fastify (proxy)
│   └── src/
│       ├── app.js
│       └── routes/
│           ├── training.routes.js
│           ├── results.routes.js
│           ├── logs.routes.js
│           ├── semantic.routes.js
│           └── benchmark.routes.js  # ★ Novo
│
├── ml-service/                      # FastAPI + PyTorch
│   └── app/
│       ├── main.py                  # Endpoints REST
│       ├── train_local.py           # ★ Script de treinamento real
│       ├── core/
│       │   ├── config.py            # Paths e env vars
│       │   ├── model_utils.py       # VAE / AE architectures
│       │   └── image_utils.py       # ★ SSIM, métricas, datasets
│       └── training/
│           └── orchestrator.py      # Simulação FedAvg
│
├── volumes/
│   ├── ml_data/                     # Datasets + pesos .pth + logs
│   └── resultados/                  # Experimentos persistidos
│
├── paper/                           # Artigo LaTeX (IEEEtran)
└── referencias/                     # Papers de referência
```

---

## 📊 Resultados Esperados

Com modelos treinados (5–10 épocas, CPU):

| Dataset | Modelo | PSNR típico | SSIM típico | Compressão (Int8) |
|---|---|---|---|---|
| MNIST | CNN-VAE | ~20–25 dB | ~0.85–0.95 | ~87× |
| Fashion-MNIST | CNN-VAE | ~18–23 dB | ~0.80–0.92 | ~87× |
| CIFAR-10 | CNN-VAE | ~15–20 dB | ~0.60–0.80 | ~341× |

> CIFAR-10 tem compressão maior mas qualidade mais baixa — esperado para um modelo
> simples sem treinamento federado real. Evidencia o trade-off compressão × qualidade.

---

## ⚠️ Limitações Conhecidas

1. **Treinamento federado simulado:** O orquestrador usa pesos aleatórios para velocidade de demo. As métricas reais vêm de `train_local.py`.
2. **Sem FedAvg real:** Gradientes reais não são comunicados entre nodos — isso requereria Flower (flwr) ou framework similar.
3. **Sem métrica de acurácia semântica:** A acurácia por classificador (antes vs depois da compressão) não está implementada — usar SSIM como proxy.
4. **CPU-only:** O serviço não usa GPU no Docker por padrão. Treinar com GPU requer `runtime: nvidia` no docker-compose.

---

## 🔮 Trabalhos Futuros

- Integrar **Flower (flwr)** para FedAvg com gradientes reais
- Adicionar **classificador pré-treinado** para métrica de acurácia semântica end-to-end
- Implementar **compressão Top-K de gradientes** para comunicação eficiente
- Avaliar com **ViT (Vision Transformer)** como encoder semântico
- Adicionar **canal físico realista** (fading, interferência) via GNU Radio

---

## 📜 Contexto Acadêmico

Projeto de Iniciação Científica (IC) —
**Universidade Tecnológica Federal do Paraná (UTFPR)**, Câmpus Cornélio Procópio.

Tema: Comunicação semântica eficiente via aprendizado federado e codificação latente.