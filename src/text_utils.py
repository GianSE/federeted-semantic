import torch

def tokenize_text(text, size=10):
    """Converte Texto -> Tensor (Normalizado 0 a 1)"""
    # Pega os primeiros 'size' caracteres
    vals = [ord(c) for c in text[:size]]
    # Preenche com zeros (padding) se for menor
    while len(vals) < size:
        vals.append(0)
    
    # Normaliza (ASCII vai de 0 a 255)
    tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0) / 255.0
    return tensor

def detokenize_text(tensor):
    """A MÁGICA: Converte Tensor -> Texto Legível"""
    # Desfaz a normalização (* 255)
    vals = (tensor.squeeze() * 255.0).int().tolist()
    
    # Reconstrói a string, ignorando zeros e sujeira negativa
    chars = []
    for v in vals:
        if v > 31 and v < 127: # Apenas caracteres imprimíveis ASCII
            chars.append(chr(v))
            
    return "".join(chars)

# Mantido para compatibilidade se algo antigo chamar
def text_to_tensor(text):
    return tokenize_text(text)