# 🧠 Federated Learning com Compressão Semântica (GenIA)

> Um sistema de Aprendizado Federado resiliente a falhas de rede, utilizando Autoencoders para reconstrução semântica de dados perdidos.

![Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Python](https://img.shields.io/badge/Python-3.9-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## 📋 Sobre o Projeto

Este projeto demonstra uma arquitetura de **Federated Learning (FL)** onde clientes treinam um modelo de IA localmente e enviam apenas os pesos para um servidor central.

A inovação principal é o módulo **GenIA**, que permite que clientes em redes instáveis ("Client Noisy") enviem dados comprimidos ou incompletos. O servidor utiliza técnicas de reconstrução semântica para preencher as lacunas antes da agregação global.

### 🏗️ Arquitetura

O sistema roda inteiramente em **Docker** e consiste em 5 containers:

1.  **🧠 Server (Flask):** O "Cérebro". Recebe pesos, reconstrói dados faltantes (Inpainting/GenIA) e agrega o modelo global (FedAvg).
2.  **🔵 Client Full:** Cliente com conexão perfeita. Treina e envia os pesos completos.
3.  **🟠 Client Noisy:** Cliente com conexão ruim (simulada). Aplica compressão semântica (envia apenas 50% dos dados) para economizar banda.
4.  **📉 Chaos Injector:** Container privilegiado que injeta falhas reais de rede (Packet Loss, Delay) na interface do *Client Noisy* usando `tc` (Traffic Control).
5.  **🛰️ Dashboard (Streamlit):** Painel de controle para monitorar logs, métricas em tempo real e interagir com a IA.

## 🚀 Como Rodar

### Pré-requisitos
* Docker e Docker Compose instalados.

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/federated-genia-demo.git](https://github.com/SEU_USUARIO/federated-genia-demo.git)
    cd federated-genia-demo
    ```

2.  **Inicie o ambiente:**
    ```bash
    docker-compose up --build
    ```

3.  **Acesse o Dashboard:**
    Abra seu navegador em: **[http://localhost:8501](http://localhost:8501)**

## 🕹️ Como Usar (Workflow)

O sistema inicia em modo **PAUSED** para evitar treino com dataset vazio.

1.  **Ensinar (Teacher Forcing):**
    * No Dashboard, vá na barra lateral "📚 Ensinar a IA".
    * Digite uma frase correta (Ex: `Federated`) e clique em **Salvar**.
    * *Adicione algumas variações para melhorar o treino.*

2.  **Treinar:**
    * Clique no botão **▶️ INICIAR** na barra lateral.
    * Acompanhe os terminais:
        * Os clientes vão baixar o Dataset, treinar localmente e enviar ao servidor.
        * O servidor vai agregar e salvar o `global_model.pth`.
    * Veja o gráfico de **Loss** caindo (o aprendizado acontecendo).

3.  **Testar (Correção Semântica):**
    * Clique em **⏸️ PAUSAR**.
    * Vá na área "🧪 Teste de Correção".
    * Digite uma palavra com erro (Ex: `Federeted`).
    * Clique em **Verificar**.
    * A IA tentará reconstruir a palavra baseada no que aprendeu (Esperado: `Federated`).

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9
* **Machine Learning:** PyTorch (Autoencoder Semântico)
* **Comunicação:** API REST (Flask)
* **Monitoramento:** Streamlit + SQLite
* **Infraestrutura:** Docker & Docker Compose
* **Rede:** `iproute2` (Traffic Control) para injeção de falhas.

## 📂 Estrutura de Arquivos

```text
federated-genia-demo/
├── docker/
│   ├── docker-compose.yml        # 🐳 Orquestração dos 5 containers
│   ├── Dockerfile                # 🐳 Imagem base Python (ML + API + Dashboard)
│   └── requirements.txt          # 📦 Dependências (Torch, Flask, Streamlit)
│
├── src/
│   ├── core/                     # 🧠 Núcleo de ML e Federated Learning
│   │   ├── model_utils.py        # Arquitetura do Autoencoder Semântico (PyTorch)
│   │   ├── fedavg.py             # Algoritmo de agregação Federada (FedAvg)
│   │   ├── compression.py        # Compressão semântica (drop / latent / mask)
│   │   └── text_utils.py         # Conversão Texto ↔ Tensor
│   │
│   ├── server/                   # 🧠 Servidor Central (Agregador)
│   │   ├── server.py             # API Flask (upload, download, reconstrução)
│   │   └── state.py              # Controle de ciclos, status e sincronização
│   │
│   ├── client/                   # 🟠🔵 Clientes Federados
│   │   ├── client.py             # Lógica de treino local + envio de parâmetros
│   │   └── node_config.py        # Configuração (Full / Noisy / Compressão)
│   │
│   ├── dashboard/                # 🛰️ Painel de Observabilidade
│   │   ├── dashboard.py          # Interface Streamlit
│   │   └── charts.py             # Gráficos (loss, banda, latência)
│   │
│   ├── chaos/                    # 📉 Injeção de falhas de rede
│   │   └── chaos_injector.sh     # Script tc (delay, loss, bandwidth)
│   │
│   ├── storage/                  # 💾 Artefatos gerados (runtime)
│   │   ├── dataset.txt           # Frases ensinadas pelo usuário
│   │   ├── global_model.pth      # Modelo global agregado
│   │   ├── metrics.db            # Banco SQLite de métricas
│   │   └── status.json           # Estado do sistema (RUNNING / PAUSED)
│   │
│   └── logs/                     # 📜 Logs de execução (gerado)
│
├── .gitignore                    # Arquivos ignorados pelo Git
├── README.md                     # 📘 Documentação do projeto
└── LICENSE                       # (Opcional) Licença do projeto
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