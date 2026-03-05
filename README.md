# 🛰️ Aprendizado Federado para Comunicação Semântica de Imagens

> Testbed containerizado que combina Aprendizado Federado (FedAvg) com um Autoencoder Convolucional para compressão semântica de imagens, incluindo injeção de caos de rede em tempo real.

![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

---

## 📋 Sobre o Projeto

Este projeto implementa um sistema distribuído de **comunicação semântica de imagens** treinado via **Aprendizado Federado**. A ideia central é:

1. Vários dispositivos (simulados por contêineres Docker) colaboram no treinamento de um **autoencoder convolucional**
2. O autoencoder comprime imagens MNIST de **784 pixels → 32 valores** no espaço latente (redução de ~96%)
3. O vetor latente é transmitido ao receptor, que **reconstrói a imagem completa** a partir dessa representação compacta
4. Um mecanismo de **injeção de caos** (`tc netem`) simula condições adversas de rede para testar robustez

### Principais Funcionalidades

- **Compressão Semântica**: Envie apenas 4% dos dados e reconstrua a imagem no destino
- **Completação de Imagens**: Envie apenas parte da imagem e o modelo completa o restante
- **Treinamento Federado**: Clientes treinam localmente e compartilham apenas os pesos do modelo (nunca os dados)
- **Compressão Top-K**: O cliente instável envia apenas 40% dos pesos (60% de esparsidade)
- **Injeção de Caos**: Controle em tempo real de perda de pacotes, latência, corrupção e duplicação
- **Dashboard Interativo**: Painel Streamlit com visualização, testes e métricas

---

## 🏗️ Arquitetura

O sistema é composto por **5 contêineres** Docker orquestrados via Docker Compose:

| Contêiner | Função |
|---|---|
| **🧠 fl-server** | Servidor Flask — agregação FedAvg, endpoints `/reconstruct` e `/complete` |
| **🔵 client-full** | Cliente estável — treina e envia pesos completos |
| **🟠 client-noisy** | Cliente instável — aplica compressão Top-K (60% esparsidade) |
| **📊 dashboard** | Painel Streamlit — visualização, testes de comunicação e controle de caos |
| **💥 chaos-injector** | Alpine + `tc netem` — injeta falhas de rede no cliente instável |

```
┌─────────────┐     HTTP/REST      ┌─────────────┐
│ client-full │ ◄────────────────► │  fl-server  │
│  (estável)  │   pesos completos  │  (FedAvg)   │
└─────────────┘                    └──────┬──────┘
                                          │
┌─────────────┐     HTTP/REST      ┌──────┴──────┐
│client-noisy │ ◄────────────────► │  fl-server  │
│ (instável)  │   pesos Top-K      │             │
└──────┬──────┘                    └─────────────┘
       │ network_mode
┌──────┴──────┐                    ┌─────────────┐
│   chaos-    │                    │  dashboard  │
│  injector   │                    │ (Streamlit) │
│ (tc netem)  │                    │  :8501      │
└─────────────┘                    └─────────────┘
```

---

## 🚀 Como Rodar

### Pré-requisitos

- [Docker](https://www.docker.com/) e Docker Compose instalados

### Passo a Passo

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/federeted-semantic.git
cd federeted-semantic

# 2. Construa e inicie todos os contêineres
cd docker
docker compose up --build -d

# 3. Acesse o dashboard
# Abra no navegador: http://localhost:8501
```

### Comandos Úteis

```bash
# Ver logs em tempo real
docker compose logs -f

# Parar tudo
docker compose down

# Reconstruir após alterações no código
docker compose up --build -d
```

---

## 🕹️ Como Usar

### 1. Iniciar o Treinamento

- Acesse **http://localhost:8501**
- Clique em **▶️ INICIAR** na barra lateral
- Os clientes começam a treinar automaticamente com o dataset MNIST
- Acompanhe a evolução na aba **📊 Métricas**

### 2. Testar a Comunicação Semântica

Na aba **📡 Comunicação**:
- Clique em **🎲 Gerar Nova Imagem**
- Veja lado a lado: imagem original → vetor latente (32 valores) → reconstrução
- Métricas MSE e PSNR são exibidas automaticamente

### 3. Testar Completação de Imagens

Na aba **🧩 Completação**:
- Escolha o tipo de máscara (metade inferior, pixels aleatórios, metade direita)
- Ajuste a porcentagem de mascaramento (10% a 90%)
- O modelo tenta reconstruir a imagem completa a partir da parte enviada

### 4. Injetar Caos de Rede

Na barra lateral, seção **🔥 Controle de Caos**:
- **Perda de Pacotes** (0–5%)
- **Latência** (0–2000 ms)
- **Corrupção** (0–2%)
- **Duplicação** (0–5%)
- Clique em **⚡ Aplicar Caos** — as condições afetam apenas o cliente instável

---

## 🧠 Modelo: Autoencoder Convolucional

```
Encoder                              Decoder
───────                              ───────
Input (1×28×28)                      Latent (32)
    ↓ Conv2d(1→32) + ReLU                ↓ Linear(32→256)
    ↓ MaxPool(2×2)                       ↓ Linear(256→3136)
    ↓ Conv2d(32→64) + ReLU              ↓ Reshape(64×7×7)
    ↓ MaxPool(2×2)                       ↓ ConvTranspose2d(64→32) + ReLU
    ↓ Flatten(3136)                      ↓ ConvTranspose2d(32→1) + Sigmoid
    ↓ Linear(3136→256)              Output (1×28×28)
    ↓ Linear(256→32)
Latent (32)
```

| Métrica | Valor |
|---|---|
| Entrada | 784 pixels (28×28) |
| Espaço latente | 32 dimensões |
| Compressão | ~96% (784 → 32) |
| Fator | 24.5× |
| Loss | MSE (Mean Squared Error) |
| Otimizador | Adam (lr=0.001) |

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|---|---|
| ML Framework | PyTorch 2.x |
| Dataset | MNIST (60k train / 10k test) |
| Servidor | Flask (API REST) |
| Dashboard | Streamlit |
| Métricas | SQLite |
| Infraestrutura | Docker + Docker Compose |
| Caos de Rede | `tc netem` (iproute2) |

---

## 📂 Estrutura do Projeto

```
federeted-semantic/
├── docker/
│   ├── docker-compose.yml    # Orquestração dos 5 contêineres
│   ├── Dockerfile            # Imagem base (Python 3.9 + PyTorch)
│   └── requirements.txt      # Dependências Python
│
├── src/
│   ├── model_utils.py        # Autoencoder Convolucional (encoder + decoder)
│   ├── image_utils.py        # Carregamento MNIST, mascaramento, métricas
│   ├── server.py             # Servidor Flask (FedAvg + /reconstruct + /complete)
│   ├── client.py             # Cliente FL (treino local + upload de pesos)
│   ├── main.py               # Entry point (roteia para server ou client)
│   ├── dashboard.py          # Painel Streamlit (4 abas)
│   ├── chaos_loop.sh         # Loop de injeção de caos (tc netem)
│   ├── chaos_config.txt      # Configuração de caos (runtime)
│   └── logs/                 # Logs de execução dos contêineres
│
├── paper/
│   ├── main.tex              # Artigo LaTeX (IEEEtran, português)
│   ├── ref.bib               # Referências bibliográficas
│   └── acronym.tex           # Lista de acrônimos
│
├── .gitignore
└── README.md
```

---

## 📄 Artigo Científico

O diretório `paper/` contém o artigo em formato IEEE (LaTeX) documentando o sistema. Para compilar:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📜 Licença

Este projeto é parte de uma iniciação científica.
```

## ⚠️ Troubleshooting

Se precisar reiniciar do zero (limpar banco de dados e modelos salvos):

```bash
docker-compose down
# No Linux/Mac:
rm src/metrics.db src/global_model.pth src/dataset.txt src/status.json
# No Windows (PowerShell):
rm src/metrics.db, src/global_model.pth, src/dataset.txt, src/status.json
docker-compose up --build