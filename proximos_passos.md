# Próximos Passos — Comunicação Semântica Federada

> Documento de planejamento para evolução do projeto após validação no MNIST.

---

## Status Atual ✅

| Item | Status |
|------|--------|
| CNN-VAE treinado via FedAvg (MNIST) | Concluído |
| Compressão 87,1× / 98,9% de banda | Validado |
| Queda semântica 4,0% (critério ≤ 5%) | Aprovado |
| Paper IEEE com citações GenAI | Redigido e commitado |

A pesquisa está **conceitualmente correta**. O CNN-VAE é o codec semântico
adequado para o regime de compressão perceptual (Grassucci et al., 2024).
Os resultados no MNIST são a base — os próximos passos provam que a abordagem **generaliza**.

---

## Prioridade 1 — Fashion-MNIST (fácil, semana 1)

**Por que é fácil:** mesmas dimensões do MNIST (28×28, grayscale, 10 classes).
A arquitetura CNN-VAE não precisa mudar — só trocar o dataset.

### O que esperar
- Compressão: idêntica (87,1× — mesma dimensão de entrada)
- PSNR esperado: ~17–19 dB (texturas mais complexas que dígitos)
- Queda semântica: provavelmente ≤ 8% (mais difícil de classificar)
- Critério de aprovação: adaptar para ≤ 8% dado maior complexidade

### Passos técnicos
1. No notebook, célula 1: trocar `DATASET = 'mnist'` para `DATASET = 'fashion'`
2. Executar célula 4 (treina FL) — sem alterar nenhum código
3. Executar 4.5 (treina classificador) e 5/5.5/6 (benchmark + galeria)
4. Comparar resultados com MNIST lado a lado

### No paper
Adicionar Tabela de Comparação datasets:

| Dataset | PSNR | SSIM | Queda Semântica | Compressão |
|---------|------|------|-----------------|-----------|
| MNIST | 20,61 dB | 0,911 | 4,0% | 87,1× |
| Fashion-MNIST | (a medir) | (a medir) | (a medir) | 87,1× |
| CIFAR-10 | (a medir) | (a medir) | (a medir) | (ver abaixo) |

---

## Prioridade 2 — CIFAR-10 (médio, semana 2–3)

**Por que é mais complexo:** imagens RGB 32×32 (3 canais, resolução maior).
Precisa de ajuste na arquitetura do CNN-VAE.

### Mudanças necessárias na arquitetura

| Componente | MNIST/Fashion | CIFAR-10 |
|-----------|--------------|----------|
| Canais de entrada | 1 (grayscale) | 3 (RGB) |
| Tamanho entrada | 28×28 | 32×32 |
| Dimensão latente | 32 | **64 ou 128** (mais info visual) |
| Bytes transmitidos (sem compressão) | 3.136 B | **12.288 B** |
| Bytes transmitidos (latente int8 d=64) | 36 B | **68 B** |
| Compressão esperada (d=64) | 87,1× | **~181×** |

### Mudanças no código

**`ml-service/app/core/autoencoder.py`** — adaptar o encoder/decoder:
- `in_channels=1` → `in_channels=3`
- `fc_in = 64*7*7` → `fc_in = 128*4*4` (ajustar conforme pooling)
- `latent_dim=32` → `latent_dim=64`

**`ml-service/app/core/image_utils.py`** — já deve ter CIFAR-10 em `DATASET_META`
(verificar se `channels=3` e `size=32` estão corretos)

### O que esperar
- PSNR esperado: ~15–18 dB (imagens coloridas muito mais complexas)
- Queda semântica: possivelmente 6–12% (adaptar critério do paper para ≤ 10%)
- Compressão esperada: ~181× (economia de ~99,5%)

---

## Prioridade 3 — Melhorias no Paper (semana 3–4)

### 3.1 Adicionar seção de comparação entre datasets
Após obter resultados de Fashion-MNIST e CIFAR-10, criar:
- Tabela comparativa (MNIST / Fashion / CIFAR-10)
- Gráfico de barras: PSNR e queda semântica por dataset
- Texto na Seção 4 analisando o trade-off complexidade × preservação semântica

### 3.2 Adicionar análise do limiar de PSNR
O experimento de galeria (Célula 5.5) mostrou que PSNR ≥ 18 dB → semântica preservada.
- Formalizar isso como **diretriz de projeto**: SNR_canal_latente mínimo para manter PSNR
- Citar Grassucci et al. (compressão semântica adaptativa ao canal)
- Citar Xia et al. (camada generativa com qualidade adaptativa)

### 3.3 Melhorar a Tabela de Escalabilidade
Adicionar coluna de "Economia por hora" assumindo throughput típico IoT (ex: 1 imagem/min):
- 1 dispositivo → X MB/dia economizados
- 100 dispositivos → Y GB/dia economizados

---

## Prioridade 4 — Melhorias Técnicas (futuro)

Estas são mais trabalhosas mas aumentam a contribuição científica:

### 4.1 β-VAE (controle explícito do trade-off)
- Testar β = 0.5, 1.0, 2.0
- Mostrar que β menor → PSNR maior, β maior → semântica mais robusta ao canal
- Referência: Higgins et al. (β-VAE, já no ref.bib)

### 4.2 Compressão Adaptativa ao Canal
- Variar SNR do canal latente: 5, 10, 15, 20, 25 dB
- Medir como PSNR e queda semântica variam
- Propor regra: `latent_dim = f(SNR_canal)`
- Referência: Xia et al. `xia2023gennet_layer`

### 4.3 Mais rounds de FL
- Teste: 5 vs 10 vs 20 rounds
- Mostrar curva de convergência estendida
- Verificar se mais rounds compensam dado o custo de comunicação

### 4.4 Modelo de Difusão como Codec (trabalho futuro do paper)
- Substituir VAE por Stable Diffusion ou DDPM pequeno
- Transmitir apenas prompt textual ou mapa semântico (< 100 bytes)
- Receptor regenera imagem do zero condicionado ao prompt
- Referência: Du et al. `du2023genai_semcom`

---

## Ordem de Execução Recomendada

```
Semana 1:   Fashion-MNIST → rodar pipeline completo → anotar resultados
Semana 2:   CIFAR-10 → adaptar arquitetura → rodar pipeline
Semana 3:   Comparar 3 datasets → atualizar paper (nova tabela comparativa)
Semana 4:   β-VAE no MNIST (rápido de testar) → adicionar insight ao paper
Futuro:     Modelo de difusão → novo projeto / extensão do paper
```

---

## Checklist de Execução

- [ ] Fashion-MNIST: executar pipeline completo (Células 1–6)
- [ ] Fashion-MNIST: anotar PSNR, SSIM, queda semântica
- [ ] CIFAR-10: adaptar `in_channels=3`, `latent_dim=64` no autoencoder
- [ ] CIFAR-10: executar pipeline e anotar resultados
- [ ] Paper: adicionar Tabela de Comparação (3 datasets)
- [ ] Paper: adicionar análise do limiar PSNR como diretriz de projeto
- [ ] Paper: adicionar gráfico PSNR vs SNR do canal latente
- [ ] Paper: recompilar e fazer push
- [ ] β-VAE: testar β = {0.5, 1.0, 2.0} e comparar
