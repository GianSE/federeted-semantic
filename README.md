# 🛰️ Comunicação Semântica Federada IoT/6G — Testbed de Pesquisa

> Protótipo de pesquisa que demonstra como **Nós Baseados em IA Generativa** (Autoencoders Variacionais/AE)
> reduzem o volume de dados transmitidos na borda preservando apenas a informação semântica utilitária. 
> Avaliado nos datasets MNIST, Fashion-MNIST e CIFAR-10.

[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)](https://jupyter.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)

---

## 📋 Hipótese e O "Trade-off Triplo"

O Testbed foi projetado para provar que a compressão extrema através de um Extrator de Características Semânticas na transmissão, acoplado a um Nó GenAI receptor, é a arquitetura ideal na futura conectividade IoT/6G.

A análise metodológica foca no Ponto de Equilíbrio entre 3 vetores matemáticos:
1. **Compressão (Banda Reduzida)**
2. **Qualidade do Canal Físico Simulado (AWGN e Perdas)**
3. **Robustez Semântica (Acurácia Oculta pelo Classificador)**

---

## 🏗️ Arquitetura do Sistema Científico

O sistema foi redesenhado para um workflow analítico governado inteiramente via **Jupyter Notebook**, que atua como Cérebro e Dashboard consumindo APIs fechadas nos contêineres:

```text
┌─────────────────────────────────────────────────────────────────┐
│  PLATAFORMA DE EXPERIMENTAÇÃO (Jupyter Notebook)                │
│  experimento_federado.ipynb                                     │
│  Controla os laços, define complexidade e renderiza o Trade-off │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST requests
┌──────────────────────────▼──────────────────────────────────────┐
│  ORQUESTRADOR E API DE IA (ml-service)                          │
│  FastAPI + PyTorch 2.5 — Computa Modelos e simula Cenários      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Dispara Sub-Processos
┌──────────────────────────▼──────────────────────────────────────┐
│  REDE COLETIVA FEDERADA (FL-Nodes)                              │
│  [ fl-server ] <--> [ fl-client-1 ] , [ fl-client-2 ]           │
│  Simulação estrita de Federated Averaging em nós isolados       │
└─────────────────────────────────────────────────────────────────┘
              ↕ Compartilhamento I/O Otimizado
┌─────────────────────────────────────────────────────────────────┐
│  /shared_data/  — Volumes onde datasets, pesos .pth e gráficos  │
│                   são gerados, guardados e consumidos.          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Como Executar o Experimento

### Pré-requisitos
- **Docker Desktop** ou Engine ativo na máquina.
- Ambiente Python com capacidade de ler arquivos **Jupyter Notebook** (ex: *VS Code* com a extensão Jupyter instalada).

### Passo 1: Ligar a Infraestrutura Pesada
Abra o terminal na pasta raiz e instancie a fábrica central do projeto:

```bash
docker-compose up --build -d
```
*(Isso subirá as APIs FastAPI silenciosamente).*

### Passo 2: O Laboratório de Pesquisa
Abra o arquivo **`experimento_federado.ipynb`**. Ele concentrará todas as ferramentas de avaliação estatística. 

Apenas siga as células sequencialmente em um dos três módulos existentes:

- **1º Bloco - Configurações Fundamentais**: Ajuste as métricas primárias (Ex: altere para `DATASET = 'cifar10'`), confirme os contêineres e inicialize o cache.
- **2º Bloco - Auditoria Visual (Opcional)**: Células manuais úteis para treinar rapidamente o Nó Edge e testar "visualmente" a distorção injetando uma foto degradada por ruído de canal (AWGN).
- **3º Bloco - A Métrica do Paper (Célula de Trade-off)**: A ferramenta suprema que treinará automaticamente o limite iterativo da banda (`16 bytes`, `32 bytes`, etc), avaliando o sacrifício na Queda Semântica, e salvando seu Gráfico Tese no final da varredura.

---

## 📂 Estrutura de Diretórios

```text
federeted-semantic/
├── docker-compose.yml           # Infraestrura do laboratório
├── experimento_federado.ipynb   # O Console de Controle Central
│
├── ml-service/                  # O Corpo e o Cérebro de IA
│   └── app/
│       ├── main.py              # API Gateway dos Modelos PyTorch
│       ├── classifier_utils.py  # Avaliador Matemático da Queda Semântica
│       └── orchestrator.py      # Gerente do Laço FL
│
├── fl-server/                   # Nó Agregador (Federated Learning)
├── fl-client/                   # Antenas Edge Locais (Dispositivos)
│
├── shared_data/                 # Cache, Volumes Montados
│   ├── ml-data/                 # Datasets Massivos e Logs de perdas
│   └── resultados/              # Destino final dos gráficos de tese
│
├── paper/                       # Redação Oficial em LaTeX (IEEEtran)
└── referências_vetorizadas/     # Textos teóricos base para o Nó GenAI
```

---

## 📜 Contexto Acadêmico

Projeto de Ciência da Computação / Engenharia — **Universidade Tecnológica Federal do Paraná (UTFPR)**, Câmpus Cornélio Procópio.
Tema: Comunicação semântica eficiente via aprendizado federado e Nós de IA Generativos de borda.