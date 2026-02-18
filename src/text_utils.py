import torch

# Tamanho padrão fixo para facilitar LSTMs
MAX_LEN = 30 
VOCAB_SIZE = 128 # ASCII padrão

def tokenize_text(text, size=MAX_LEN):
    """Converte Texto -> Tensor de Índices Inteiros (LongTensor)"""
    # Garante tamanho máximo
    text = text[:size]
    
    # Converte chars para índices ASCII
    vals = [ord(c) if ord(c) < VOCAB_SIZE else 32 for c in text] # 32 = Espaço se for char estranho
    
    # Padding (preenche com 0)
    while len(vals) < size:
        vals.append(0)
    
    # Retorna LongTensor (Inteiros) para Embedding
    return torch.tensor(vals, dtype=torch.long).unsqueeze(0)

def detokenize_text(logits):
    """Converte Logits (Probabilidades) -> Texto"""
    if logits.dim() == 3:
        indices = torch.argmax(logits, dim=2).squeeze().tolist()
    else:
        indices = logits.tolist()

    if isinstance(indices, int): indices = [indices]
        
    chars = []
    for idx in indices:
        # Se encontrar um 0 (padding), PARE imediatamente.
        # Isso evita mostrar lixo se o modelo alucinar no final.
        if idx == 0:
            break 
            
        if idx > 0 and idx < VOCAB_SIZE:
            chars.append(chr(idx))
            
    return "".join(chars)