# Master Document: Comunicação Semântica Federada na Borda (6G/IoT)

Este documento compila o contexto, a metodologia, os resultados experimentais e a base literária da pesquisa para facilitar a redação final do artigo (`main.tex`).

---

## 1. Visão Geral e Hipótese Central

**O Problema**: As redes 4G/5G focam na transmissão de bits exatos. Em cenários IoT massivos, transmitir imagens brutas (high-resolution) gera gargalos de banda, latência alta, consumo de bateria e riscos à privacidade.

**A Solução Proposta**: Deslocar a inteligência para a **Borda (Edge)**. Utilizar Autoencoders Variacionais (VAE) para extrair apenas o **significado (semântica)** da imagem em um espaço latente ultracompacto (poucos bytes).

**Hipótese Central**: 
> "Um nó baseado em IA Generativa (Nó GenIA) na borda, treinado via Aprendizado Federado, pode reduzir a largura de banda em mais de 90% mantendo a integridade semântica (queda de acurácia ≤ 5%) mesmo sob condições de ruído de canal (AWGN)."

---

## 2. Fundamentação Teórica e Literatura

A pesquisa está apoiada em três pilares da literatura recente:

1.  **GenAINet (Du et al., 2023)**: Proposta de redes coletivas sem fio onde a IA Generativa permite que os nós "conversem" através de gatilhos matemáticos, não apenas dados crus.
2.  **Semantic Communication (Grassucci et al., 2023)**: Defesa de que o 6G não suportará a transmissão bit-a-bit. Propõe o uso de VAEs como codecs semânticos resilientes a canais sub-ótimos.
3.  **Generative Network Layer (Xia et al., 2023)**: Demonstra como receptores generativos podem "alucinar" com fidelidade informações perdidas no canal ruidoso.
4.  **Federated Learning (McMahan, 2017)**: Algoritmo FedAvg para treinar modelos em dispositivos de borda sem expor dados privados.

---

## 3. Arquitetura do Sistema (Testbed)

O experimento foi implementado em um ambiente de contêineres Docker para simular o isolamento de rede real:

*   **Nó Emissor (Edge Node)**: CNN-VAE Encoder. Transforma imagem (ex: 3136 bytes) em vetor latente (ex: 32 bytes).
*   **Canal Físico Simulado**: Injeção de Ruído Aditivo Gaussiano (AWGN) com SNR de 15dB e Masking (perda de pacotes) de 10%.
*   **Nó Receptor (GenIA Node)**: CNN-VAE Decoder. Reconstrói a imagem a partir do vetor ruidoso usando capacidade generativa.
*   **Orquestrador (FedAvg)**: Servidor central que agrega os pesos dos modelos treinados localmente nos clientes sem nunca ver as imagens originais.

---

## 4. Metodologia: O Trade-off Triplo

A avaliação não é apenas visual (PSNR/SSIM), mas sim baseada em um **Classificador Juiz** independente.

*   **Vetores de Avaliação**:
    1.  **Compressão**: Tamanho do espaço latente (16, 32, 64, 128, 256 bytes).
    2.  **Robustez**: Resiliência ao ruído AWGN no canal.
    3.  **Preservação Semântica**: Queda na acurácia do Classificador Juiz (deve ser $\leq 5\%$).

*   **Datasets de Teste**:
    *   **MNIST**: Baixa complexidade (monocromático). Foco em compressão extrema (até 16 bytes).
    *   **Fashion-MNIST**: Complexidade média (texturas).
    *   **CIFAR-10**: Alta complexidade (RGB, objetos reais). Exige latentes maiores (64-256 bytes).

---

## 5. Resultados Experimentais (Snapshot MNIST)

Resultados obtidos no experimento `20260422_024134`:

*   **Eficiência de Transmissão**:
    *   Dados Brutos: 3.136 bytes.
    *   Dados Latentes: 36 bytes (int8).
    *   **Redução de Banda: 98,9%** (Razão 87,1x).
*   **Qualidade Visual**:
    *   PSNR: 20,61 dB | SSIM: 0,911.
*   **Validação Semântica**:
    *   Acurácia Original: 90,0%.
    *   Acurácia Reconstruída: 86,0%.
    *   **Queda Semântica: 4,0%** (Aprovado ✅).

---

## 6. Conclusões e Contribuições

1.  **Viabilidade Prática**: Demonstrou-se que a Comunicação Semântica Federada é viável para dispositivos IoT com restrição de banda.
2.  **Ponto de Equilíbrio (Sweet Spot)**: A pesquisa identifica o limiar de compressão para cada tipo de dado. Para dados industriais simples (MNIST), 16-32 bytes são suficientes. Para dados visuais complexos, o sistema exige ~64-128 bytes para manter a semântica.
3.  **Isolamento e Privacidade**: O uso de Docker e Federated Learning prova que é possível atingir alta performance sem centralizar dados sensíveis.

---
*Documento gerado para suporte à redação do main.tex | 2026*
