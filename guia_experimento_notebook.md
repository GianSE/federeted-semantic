# Guia do Experimento: Trade-off de Comunicação Semântica

Este documento descreve a estrutura do seu `experimento_federado.ipynb` e como utilizá-lo para provar a hipótese central da sua pesquisa.

## 🎯 A Hipótese Central (O Seu "Gráfico 3D")

Seu projeto consiste em provar uma relação de custo-benefício tripartite:
1. **Compressão (Dimensão Latente)**: O quão pequena a imagem se torna.
2. **Economia de Banda**: O impacto real de transmitir menos bytes na rede.
3. **Perda de Semântica**: O quanto a imagem se degradou na visão de um "robô" (classificador), e não apenas do olho humano.

Como os dados possuem diferentes complexidades (ex: *MNIST* é muito mais simples que *CIFAR-10*), a taxa de compressão aceitável varia. O seu objetivo ao usar este Notebook é **encontrar o "Sweet Spot" (Ponto de Equilíbrio) para cada dataset**, respondendo à pergunta: *"Até onde posso esmagar o vetor para salvar banda sem destruir o significado da imagem?"*

---

## 🛠️ Os Módulos do `experimento_federado.ipynb`

O seu Notebook agora está dividido em módulos funcionais. Para a sua meta de levantar os gráficos de Trade-off (Dataset por Dataset), **você só precisa utilizar o Módulo 1 e o Módulo 3**.

### Módulo 1: As Fundações (Células 1, 2 e 3)
*Obrigatório. Levanta a infraestrutura básica e os dados brutos dependendo do dataset que você deseja analisar no momento.*

*   **Célula 1 (Setup Global)**: É a matriz do seu teste. Você altera a variável `DATASET` para apontar o dado do dia (ex: `'cifar10'` hoje, `'mnist'` amanhã).
*   **Célula 2 (Checagem)**: Envia pings para confirmar que o Docker (fl-server e ml-service) está vivo e respirando.
*   **Célula 3 (Integração de Dados)**: Verifica se o `DATASET` escolhido na Célula 1 já foi fisicamente baixado para a memória dos containers.

### Módulo 2: O Laboratório Visual (Células 4, 4.5 e 5)
*Opcional e voltado a "Provas Qualitativas".*

*Esse módulo serve exclusivamente para gerar imagens para os seus slides de apresentação. Ele não gera a curva gráfica da tese, mas gera evidências reais.* Se você quiser ver como um passarinho do CIFAR-10 de **32 bytes** fica no final da transmissão, você roda este módulo.

### Módulo 3: O Gerador da Tese (A Nova Célula de Trade-off)
*Obrigatório. Esta é a "Célula Mágica" que constrói as evidências científicas que você procura.*

*   **A "Nova" Célula 6 (Análise de Trade-off)**: Ela lê a complexidade do `DATASET` que você marcou na Célula 1, e entra em um *loop de experimentação destrutivo*. Ela treinará e reconstruirá as redes autonomamente para `[16, 32, 64, 128]` dimensões. No final de cada rodada, ela anota o tamanho da Economia de Banda gerada e pede ao Classificador para cruzar com a Perda Semântica.
* **O Resultado Final**: Gera e salva o gráfico de Ponto de Equilíbrio. **É este gráfico que provará para um Dataset Y, a compressão aceitável é X.**

---

## 🚀 Como gerar sua Tese (Passo a Passo)

Para construir o seu TCC, você repetirá o ciclo abaixo **três vezes** (uma para MNIST, uma para Fashion, uma para CIFAR-10):

1. Vá na **Célula 1** e altere `DATASET = 'mnist'`
2. Rode as Fundações (Células 1, 2 e 3).
3. Vá direto para o **Módulo 3** (Análise de Trade-off) e rode a célula.
4. Salve a imagem do gráfico gerado.
5. Repita o processo alterando a Célula 1 para `'fashion'` e finalmente para `'cifar10'`.

Ao final, você terá 3 gráficos provando matematicamente quando e como ativar a compressão em uma rede IoT/6G dependendo da textura e peso do dado transmitido.
