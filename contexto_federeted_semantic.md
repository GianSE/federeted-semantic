# Contexto da Pesquisa: Comunicação Semântica Federada 

Este documento formaliza toda a essência arquitetural, teórica e prática do seu Trabalho de Conclusão de Curso (TCC) / Pesquisa Científica envolvendo Inteligência Artificial nas redes de telecomunicação futuras (IoT/6G). 

---

## 1. A Grande Ideia (O Problema e a Hipótese)

Nas redes sem fio tradicionais (como 4G/5G), a comunicação foca na transmissão impecável de "bits". Se um sensor remoto fotografa algo em campo, ele tenta transmitir a imagem inteira pela rede de banda estreita. Isso drena a bateria, congestiona o canal e sofre alta latência. 

No entanto, para redes do futuro (6G/IoT massivo), a máquina que recebe os dados quase sempre *não precisa* da figura inteira em 4K. Ela só precisa extrair o **significado** ou a **tarefa** (ex: "é um pássaro" ou "é um caminhão"). Entramos então na **Comunicação Semântica**:

**A Hipótese Central:**
*"Em vez de transmitir milhares de pixels crus (matrizes pesadas), uma antena pode usar um Autoencoder Dinâmico para traduzir a foto num Espaço Latente de dezenas de bytes. Ao receber esse resumo ultracompacto, a antena remota fará o decode dos pixels essenciais. O sistema vai provar que eu posso economizar mais de 90% da Banda do Canal de Telecomunicação mantendo a queda de percepção de uma máquina abaixo de um limite aceitável de 10% (Tolerância de Trade-off)."*

---

## 2. O Que Foi Implementado no Testbed (A Engenharia)

Seu projeto não é apenas uma teoria, ele foi montado sob um Testbed robusto replicando a vida real:

### A Arquitetura Híbrida e o Paradigma "Nó GenIA":
1. **Inteligência Federada (`fl-server` e `fl-clients`)**: Diferente das abordagens centralizadas, o projeto instaura o conceito de "Redes Coletivas Sem Fio Baseadas em IA Generativa" (*GenAINet: Enabling Wireless Collective*). Através do Federated Learning, os dispositivos IoT são convertidos em nós inteligentes ("Nós Edge") que treinam redes mutuamente sem exportar dados brutos invasivos.
2. **O Nó Receptor GenIA (Camada Generativa)**: Este é o elo direto da sua pesquisa com o estado-da-arte (*Generative Network Layer for Communication* e *Generative AI Meets Semantic Communication*). Ao invés do receptor simplesmente descompactar um arquivo tradicional (zip), ele atua como um **Nó GenIA**. A antena emissora envia um "gatilho" hiper-sintético na nuvem (entre 16 a 128 bytes). Ao chegar na base, a inteligência generativa alojada no Decoder alucina e ressintetiza a física intrínseca da imagem original apenas pelo contexto recebido. Em suma, trocamos largura de banda pesada por processamento computacional inteligente no receptáculo.
3. **O Sistema "Juiz" Classificador**: Ao invés de uma métrica puramente visual (como o humano vendo pixels em SSIM/PSNR), o sistema tem um Robô Classificador paralelo. Ele dita os limites estatísticos, acusando se a mensagem do Nó GenIA, após a geração sintética, ainda detém as tarefas semânticas pretendidas ou se distorceu os traços de um gato em um cachorro.
4. **Variáveis de Canal Físico Simulado**: O `ml-service` foi construído aceitando ruídos do tipo Aditivo Gaussiano (AWGN) e Perda de Pacotes por Máscara, testando a capacidade intrínseca do Nó GenIA reconstruir contextos falhos no vácuo radiante.

---

## 3. O Escopo Focado de Laboratório (O Trade-off Triplo)

O diferencial crucial de sua pesquisa que prova maturidade é assumir que **não existe um compressor universal bom para todos os cenários**. A semântica exige arquiteturas distintas consoantes à riqueza fotográfica.
A avaliação se baseia em "Trade-off":
**`Economia Transmitida (Banda)` vs `Fator de Compressão` vs `Taxa de Sangria Semântica (Acurácia Oculta)`**.

Testando a varredura progressiva sobre:
- **MNIST:** (Filtros Simples Unicanal - Números Brancos em Tela Preta) 
- **Fashion-MNIST:** (Texturas simples - Vestuários)
- **CIFAR-10:** (Densidade caótica 3 canais - Animais e Veículos super detalhados)

---

## 4. Resultados Esperados

Ao rodar a Automação Analítica e colher os dados gráficos unificados que o Jupyter Notebook cospe, espera-se poder redigir na conclusão os seguintes fatos:

1. **A Prova de Viabilidade Semântica**: Exibir que é perfeitamente viável enviar representações abstratas muito pequenas economizando entre 80~95% de Banda em dispositivos de borda IoT sem corromper as tomadas de decisões dos nós mestres (Mantendo Acertos Originais de Classificações $~90\%$).
2. **Curva de Queda Inflexível (Sweet Spot)**: O Gráfico Prossional esperado é um plano visual em cruz, na qual você provará matematicamente (sem "achismo") o Limiar ótimo para os padrões corporativos, por exemplo:
   * *"Para o modelo caótico do CIFAR-10, baixar o latente menor que 64 bytes causa um abismo na curva cruzando a perda de Tolerância Semântica > 10%. Logo, o codec 64b é o Ponto de Equilíbrio Oficial para Sensores Visuais Urbanos Complexos na Arquitetura 6G proposta."* 
   * *"Para as matrizes planas do MNIST em chãos de fábrica de contadores simples métricos, a dimensão tolera ser brutalizada para 16 bytes cravados conservando estabilidade."*

Esta obra entregará uma prova madura, com amparo estatístico e software tangível, pronta para apresentação de defesa na UTFPR.
