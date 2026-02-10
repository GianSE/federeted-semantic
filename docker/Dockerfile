FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y iproute2 procps && rm -rf /var/lib/apt/lists/*

# Copia os requisitos a partir da pasta docker
COPY docker/requirements.txt .

# Atualiza o pip primeiro para garantir compatibilidade e usa extra-index-url
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    
# Depois instala o resto (pandas, flask, etc) que já vai detectar o torch instalado
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte da pasta src para a raiz do WORKDIR (/app)
COPY src/ . 

CMD ["python", "main.py"]