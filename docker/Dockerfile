FROM python:3.9-slim

WORKDIR /app

# Ferramentas de rede (tc netem para injeção de caos)
RUN apt-get update && apt-get install -y iproute2 procps && rm -rf /var/lib/apt/lists/*

# PyTorch CPU para o pipeline textual
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Dependências do projeto
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fonte
COPY src/ .

CMD ["python", "main.py"]
