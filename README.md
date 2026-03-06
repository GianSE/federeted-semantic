# 🛰️ Aprendizado Federado para Comunicação Semântica de Imagens

> Testbed containerizado que combina Aprendizado Federado (FedAvg) com um Autoencoder Convolucional para compressão semântica de imagens, incluindo injeção de caos de rede em tempo real.

![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

---

## 📋 Sobre o Projeto

Este projeto implementa um sistema distribuído de **comunicação semântica de imagens** treinado via **Aprendizado Federado**. A ideia central é:

1. Três dispositivos (simulados por contêineres Docker) colaboram no treinamento de um **autoencoder convolucional**
2. O autoencoder comprime imagens MNIST de **784 pixels → 32 valores** no espaço latente (redução de ~96%)
3. O vetor latente é transmitido ao receptor, que **reconstrói a imagem completa** a partir dessa representação compacta
4. Um mecanismo de **injeção de caos** (`tc netem`) simula condições adversas de rede para testar robustez

### Principais Funcionalidades

- **Compressão Semântica**: Envie apenas 4% dos dados e reconstrua a imagem no destino
- **Completação de Imagens**: Envie apenas parte da imagem e o modelo completa o restante
- **Treinamento Federado**: 3 clientes com perfis distintos treinam localmente e compartilham apenas os pesos do modelo (nunca os dados)
- **Compressão Top-K**: O cliente instável envia apenas 40% dos pesos (60% de esparsidade)
- **Injeção de Caos**: Controle em tempo real de perda de pacotes, latência, corrupção e duplicação via `tc netem`
- **Dashboard Interativo**: Painel Streamlit com 6 abas (Arquitetura, Comunicação, Completação, Terminais, Métricas, Experimentos)
- **Automação de Experimentos**: Script para execução automática de 4 cenários de caos com geração de figuras e tabelas LaTeX
- **Timeout de Agregação**: Servidor agrega com clientes disponíveis após 30s, garantindo progresso mesmo sob falhas

---

## 🏗️ Arquitetura

O sistema é composto por **7 contêineres** Docker orquestrados via Docker Compose:

| Contêiner | Função |
|---|---|
| **🧠 fl-server** | Servidor Flask — agregação FedAvg com timeout, endpoints `/reconstruct`, `/complete` e `/reset_round` |
| **🔵 client-full** | Cliente estável — dados IID, envia pesos completos |
| **🟠 client-noisy** | Cliente instável — aplica compressão Top-K (60% esparsidade), alvo do injetor de caos |
| **🟢 client-noniid** | Cliente heterogêneo — treina apenas com dígitos 0–3 (Non-IID) |
| **📊 dashboard** | Painel Streamlit (porta 8501) — 6 abas de visualização, testes e controle |
| **💥 chaos-injector** | Alpine + `tc netem` — injeta falhas de rede no cliente instável |
| **🔧 runner** | Serviço utilitário (profile `tools`) — executa experimentos e gera figuras/tabelas |

```
┌──────────────┐    HTTP/REST     ┌──────────────┐
│  client-full │ ◄──────────────► │              │
│  (estável)   │  pesos completos │              │
└──────────────┘                  │              │
                                  │   fl-server  │
┌──────────────┐    HTTP/REST     │   (FedAvg)   │
│ client-noisy │ ◄──────────────► │              │
│  (instável)  │  pesos Top-K     │  timeout 30s │
└──────┬───────┘                  │              │
       │ network_mode             │              │
┌──────┴───────┐    HTTP/REST     │              │
│    chaos-    │  ┌──────────────►│              │
│   injector   │  │               └──────────────┘
│  (tc netem)  │  │
└──────────────┘  │               ┌──────────────┐
                  │               │   dashboard  │
┌──────────────┐  │               │ (Streamlit)  │
│client-noniid │ ─┘               │   :8501      │
│  (Non-IID)   │                  └──────────────┘
│ dígitos 0–3  │
└──────────────┘
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
docker compose -f docker/docker-compose.yml up --build -d

# 3. Acesse o dashboard
# Abra no navegador: http://localhost:8501
```

### Comandos Úteis

```bash
# Ver logs em tempo real
docker compose -f docker/docker-compose.yml logs -f

# Ver logs apenas do servidor
docker compose -f docker/docker-compose.yml logs fl-server --tail 20

# Parar tudo
docker compose -f docker/docker-compose.yml down

# Reconstruir após alterações
docker compose -f docker/docker-compose.yml up --build -d
```

---

## 🧪 Experimentos Automatizados

O projeto inclui scripts para execução automatizada de experimentos e geração de figuras/tabelas para o artigo.

### Executar os 4 Cenários de Caos

```bash
# Certifique-se de que o servidor está rodando
docker compose -f docker/docker-compose.yml up -d fl-server client-full client-noisy client-noniid chaos-injector

# Execute os experimentos (Normal, Leve, Moderado, Severo)
docker compose -f docker/docker-compose.yml run --rm runner python run_experiments.py
```

Cada cenário aplica condições crescentes de caos no cliente instável:

| Cenário | Perda | Latência | Corrupção | Duplicação |
|---|---|---|---|---|
| Normal | 0% | 0 ms | 0% | 0% |
| Leve | 1% | 200 ms | 0% | 0% |
| Moderado | 3% | 500 ms | 0,5% | 1% |
| Severo | 5% | 1000 ms | 2% | 5% |

### Gerar Figuras e Tabelas

```bash
# Gera 4 figuras PNG e 3 tabelas LaTeX a partir dos resultados
docker compose -f docker/docker-compose.yml run --rm runner python generate_figures.py
```

**Saídas geradas:**

| Arquivo | Descrição |
|---|---|
| `paper/figures/fig_convergence.png` | Curvas de convergência (loss por rodada, 3 clientes) |
| `paper/figures/fig_reconstruction.png` | Imagens originais vs. reconstruções semânticas |
| `paper/figures/fig_completion.png` | Completação de imagens parciais (3 tipos de máscara) |
| `paper/figures/fig_chaos.png` | Impacto do caos na convergência (MSE e PSNR por cenário) |
| `paper/tables/tab_reconstruction.tex` | Métricas de reconstrução por dígito (0–9) |
| `paper/tables/tab_completion.tex` | Métricas de completação por tipo/nível de mascaramento |
| `paper/tables/tab_chaos_results.tex` | Resultados comparativos dos 4 cenários de caos |

### Resultados Obtidos

Os resultados dos experimentos ficam salvos em `src/results/`:

| Métrica | Valor |
|---|---|
| MSE médio (reconstrução) | 0,073 |
| PSNR médio | 11,6 dB |
| Melhor dígito | "1" — MSE 0,035, PSNR 14,6 dB |
| Pior dígito | "3" — MSE 0,111, PSNR 9,5 dB |
| Cenário Normal | MSE 0,071, 22 rodadas, ~3 min |
| Cenário Severo | MSE 0,068, 30 rodadas, ~22 min |
| Completação (75% mascarado) | degradação máxima de 23% no MSE |

---

## 🕹️ Como Usar o Dashboard

### 1. Iniciar o Treinamento

- Acesse **http://localhost:8501**
- Clique em **▶️ INICIAR** na barra lateral
- Os 3 clientes começam a treinar automaticamente com o dataset MNIST
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
- O modelo reconstrói a imagem completa a partir da parte enviada

### 4. Injetar Caos de Rede

Na barra lateral, seção **🔥 Controle de Caos**:
- **Perda de Pacotes** (0–5%)
- **Latência** (0–2000 ms)
- **Corrupção** (0–2%)
- **Duplicação** (0–5%)
- Clique em **⚡ Aplicar Caos** — as condições afetam apenas o cliente instável

### 5. Comparar Cenários de Caos

Na aba **🔬 Experimentos**:
- Visualize os resultados dos 4 cenários de caos lado a lado
- Compare curvas de convergência, métricas finais e tempo de treinamento

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
| Fator | 24,5× |
| Loss | MSE (Mean Squared Error) |
| Otimizador | Adam (lr=0,001) |
| Épocas locais/rodada | 5 |
| Batch size | 32 |

### Protocolo FedAvg

1. Servidor inicializa modelo global
2. A cada rodada, os 3 clientes baixam o modelo global
3. Cada cliente treina localmente por 5 épocas
4. `client-noisy` aplica compressão Top-K (mantém 40% dos pesos)
5. `client-noniid` treina apenas com dígitos 0–3
6. Servidor agrega pela média (FedAvg) — timeout de 30s para clientes faltantes
7. Repete até convergência

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|---|---|
| ML Framework | PyTorch 2.x |
| Dataset | MNIST (60k train / 10k test) |
| Servidor | Flask (API REST) |
| Dashboard | Streamlit |
| Métricas | SQLite |
| Figuras | Matplotlib + Seaborn |
| Infraestrutura | Docker + Docker Compose |
| Caos de Rede | `tc netem` (iproute2) |
| Artigo | LaTeX (IEEEtran) |

---

## 📂 Estrutura do Projeto

```
federeted-semantic/
├── docker/
│   ├── docker-compose.yml      # Orquestração dos 7 serviços
│   ├── Dockerfile              # Imagem base (Python 3.9 + PyTorch)
│   └── requirements.txt        # Dependências Python
│
├── src/
│   ├── main.py                 # Entry point (roteia para server ou client)
│   ├── server.py               # Servidor Flask (FedAvg + timeout + /reconstruct + /complete + /reset_round)
│   ├── client.py               # Cliente FL (treino local + upload de pesos + Top-K)
│   ├── config.py               # Configurações (ROUND_TIMEOUT, etc.)
│   ├── model_utils.py          # Autoencoder Convolucional (encoder + decoder)
│   ├── image_utils.py          # Carregamento MNIST, mascaramento, métricas
│   ├── log_utils.py            # Utilitários de logging
│   ├── dashboard.py            # Painel Streamlit (6 abas)
│   ├── run_experiments.py      # Automação dos 4 cenários de caos
│   ├── generate_figures.py     # Geração de figuras e tabelas LaTeX
│   ├── chaos_loop.sh           # Loop de injeção de caos (tc netem)
│   ├── chaos_config.txt        # Configuração de caos (runtime)
│   ├── global_model.pth        # Modelo global salvo
│   ├── metrics.db              # Banco SQLite com métricas
│   ├── status.json             # Estado atual do treinamento
│   ├── results/                # Resultados dos experimentos (CSVs + modelos por cenário)
│   ├── logs/                   # Logs de execução
│   └── data/MNIST/             # Dataset MNIST (baixado automaticamente)
│
├── paper/
│   ├── main.tex                # Artigo LaTeX (IEEEtran, português)
│   ├── main.pdf                # PDF compilado
│   ├── ref.bib                 # Referências bibliográficas
│   ├── acronym.tex             # Lista de acrônimos
│   ├── figures/                # Figuras geradas (4 PNGs)
│   └── tables/                 # Tabelas geradas (3 .tex)
│
├── .gitignore
└── README.md
```

---

## 📄 Artigo Científico

O diretório `paper/` contém o artigo em formato IEEE (LaTeX, IEEEtran) documentando o sistema, escrito em português. O artigo inclui:

- Descrição da arquitetura containerizada
- Protocolo de treinamento federado com FedAvg
- Análise de reconstrução semântica (MSE e PSNR por dígito)
- Testes de completação de imagens parciais
- Avaliação de resiliência com 4 cenários de injeção de caos
- Figuras e tabelas geradas automaticamente a partir dos experimentos

### Compilar o PDF

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

> **Requisito:** Distribuição LaTeX instalada (MiKTeX no Windows ou TeX Live no Linux/Mac).

---

## ⚠️ Troubleshooting

### Reiniciar do zero

```bash
docker compose -f docker/docker-compose.yml down

# Limpar dados (PowerShell)
Remove-Item src/metrics.db, src/global_model.pth, src/status.json -ErrorAction SilentlyContinue

# Limpar dados (Linux/Mac)
rm -f src/metrics.db src/global_model.pth src/status.json

# Reconstruir
docker compose -f docker/docker-compose.yml up --build -d
```

### PDF não compila

Se o PDF estiver aberto em um leitor (Acrobat, Edge, etc.), feche-o antes de recompilar — o arquivo fica bloqueado no Windows.

### Timeout do servidor

O servidor agrega automaticamente com os clientes disponíveis após 30 segundos (`ROUND_TIMEOUT=30`). Se um cliente falhar ou demorar, o treinamento continua normalmente.

### Reexecutar apenas um cenário

Para resetar a rodada entre cenários:
```bash
curl -X POST http://localhost:5000/reset_round
```

---

## 📜 Licença

Este projeto é parte de uma iniciação científica na **Universidade Tecnológica Federal do Paraná (UTFPR)** — Cornélio Procópio.