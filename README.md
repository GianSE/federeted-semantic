# Aprendizado Federado para Comunicação Semântica de Texto

Projeto containerizado para treinar e operar um sistema de comunicação semântica 100% textual com Aprendizado Federado.

## O que o projeto faz

- Treina um autoencoder textual ou VAE textual com três clientes federados.
- Comprime texto em vetores latentes por chunks.
- Reconstrói ou gera texto no destino a partir do payload semântico.
- Completa textos parciais a partir do contexto recebido.
- Injeta caos de rede no cliente instável para avaliar robustez do treinamento.

## Arquitetura

- `fl-server`: agrega pesos com FedAvg e expõe a API textual.
- `client-full`: cliente estável com dados IID.
- `client-noisy`: cliente com compressão Top-K e caos de rede.
- `client-noniid`: cliente com subconjunto textual Non-IID.
- `dashboard`: interface Streamlit para testar compressão, geração, completação e métricas.
- `chaos-injector`: aplica `tc netem` no cliente instável.
- `runner`: executa experimentos e gera figuras/tabelas para o paper.

## Endpoints principais

- `/upload_weights`: recebe pesos dos clientes.
- `/compress_text`: recebe texto bruto e devolve payload semântico por chunks.
- `/generate_text`: recebe os latentes e gera texto no destino.
- `/complete_text`: recebe texto parcial e devolve a versão completada.
- `/reconstruct`: endpoint de baixo nível para um único vetor latente.
- `/complete`: endpoint de baixo nível para uma sequência tokenizada.

## Como rodar

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Dashboard:

```text
http://localhost:8501
```

Logs:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

Parar tudo:

```bash
docker compose -f docker/docker-compose.yml down
```

## Fluxo de teste textual

1. Inicie os containers.
2. Abra o dashboard.
3. Inicie o treinamento federado.
4. Na aba de transferência, cole um texto curto ou longo.
5. O dashboard chama `/compress_text` para gerar o payload semântico e `/generate_text` para simular a geração no destino.
6. Na aba de completação, envie texto parcial para `/complete_text`.

## Experimentos

Executar cenários de caos:

```bash
docker compose -f docker/docker-compose.yml run --rm runner python run_experiments.py
```

Gerar figuras e tabelas:

```bash
docker compose -f docker/docker-compose.yml run --rm runner python generate_figures.py
```

As figuras e tabelas geradas passam a ser textuais:

- `fig_convergence.png`: convergência do treino federado.
- `fig_reconstruction.png`: exemplos de reconstrução textual por chunks.
- `fig_completion.png`: exemplos de completação textual.
- `fig_chaos.png`: impacto do caos em CE loss e accuracy.
- `tab_reconstruction.tex`: resumo textual da reconstrução.
- `tab_completion.tex`: resumo textual da completação.
- `tab_chaos_results.tex`: comparação entre cenários.

## Estrutura relevante

- `src/text_utils.py`: dataset, tokenização, chunking e reconstrução de documentos.
- `src/model_utils.py`: `TextAutoencoder` e `TextVAE`.
- `src/server.py`: API textual e agregação federada.
- `src/client.py`: treino local e envio de pesos.
- `src/dashboard.py`: interface de operação e inspeção.
- `src/train_centralized.py`: baseline centralizado textual.
- `src/generate_figures.py`: artefatos textuais para o paper.

## Observações

- O projeto trabalha com um corpus textual local embarcado por padrão.
- Textos longos são segmentados em chunks configuráveis.
- O sistema mede CE loss e accuracy de tokens como métricas principais.
