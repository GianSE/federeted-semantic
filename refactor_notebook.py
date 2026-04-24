import json

CELL_5_SOURCE = """# Célula 5 – Treinar Classificador Juiz
# Necessário para a métrica de Queda Semântica.
import time, requests

ML_SERVICE = 'http://localhost:8000'

# Configurações Dedicadas por Nível de Dificuldade do Dataset
CONFIG_TREINO = {
    'mnist':   {'epochs': 4,  'samples': 5000},   # Muito simples, treina ultra rápido
    'fashion': {'epochs': 8,  'samples': 12000},  # Médio, roupas variam em textura
    'cifar10': {'epochs': 20, 'samples': 25000}   # Complexo, requer mergulho profundo pros canais RGB
}

print("===========================================================")
print(f" Iniciando a Escola de Juízes Semânticos API ")
print("===========================================================")
print(f"Fila de Treinamento: [{DATASET}]\\n")

DS_safe = DATASET.lower().strip()
cfg = CONFIG_TREINO.get(DS_safe, {'epochs': 5, 'samples': 3000})

print(f"\\n▶ [{DS_safe.upper()}] Acordando a API do Container...")
print(f"  [PARÂMETROS] -> {cfg['epochs']} Épocas | {cfg['samples']} Imagens no Batch")
try:
    # Inicia o treino no backend em background
    start = requests.post(f'{ML_SERVICE}/classifier/train-quick',
                          json={'dataset': DS_safe, 'epochs': cfg['epochs'], 'samples': cfg['samples'], 'seed': 42}, 
                          timeout=10)
    
    if start.status_code not in [200, 202]:
        print(f"  [ERRO DA API] O servidor devolveu Erro {start.status_code}: {start.text}")
    else:
        print(f"  [STATUS] Pedido Aceito. Monitorando o progresso...")
        
        t0 = time.time()
        ultimo_ep = -1
        while True:
            try: 
                st = requests.get(f'{ML_SERVICE}/classifier/train-quick/status', timeout=5).json()
            except:
                time.sleep(3)
                continue
            
            ep = st.get('epoch', 0)
            tot = st.get('total_epochs', cfg['epochs'])
            elapsed = int(time.time() - t0)
            
            if ep != ultimo_ep and ep > 0:
                print(f'  ✓ Epoch {ep}/{tot} ({ep/max(tot,1)*100:.0f}%) concluída [{elapsed}s]')
                ultimo_ep = ep
                
            if not st.get('running'):
                if st.get('error'):
                    print(f'  [ERRO NO TREINO]: {st["error"]}')
                elif st.get('done'):
                    acc = st.get('accuracy', 0)
                    print(f'  ★ MERGULHO CONCLUÍDO em {elapsed}s | Acurácia Forte: {acc*100:.1f}%')
                else:
                    print(f'  [AVISO] Status de execução parou misteriosamente! Verifique os logs do docker.')
                break
                
            if elapsed > 7200: # 1h de timeout pros complexos
                print('  [TIMEOUT] Tempo estourou (2 Horas de espera).')
                break
            time.sleep(4)
            
except Exception as e:
    print(f"  [FALHA DE COMUNICAÇÃO HTTP]: {e}")

print("\\n✓ Rede Julgadora Profunda finalizada! Pule para a Célula de Benchmark.")
"""

try:
    with open('experimento_federado.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Convert to list of strings with newlines
    c5_lines = [line + "\\n" if i < len(CELL_5_SOURCE.splitlines()) - 1 else line for i, line in enumerate(CELL_5_SOURCE.splitlines())]

    updated_c5 = False
    updated_c6 = False

    for c in nb['cells']:
        if c['cell_type'] == 'code':
            content = "".join(c.get('source', []))
            
            # Identify Cell 5
            if "Célula 5 – Treinar Classificador" in content:
                c['source'] = c5_lines
                updated_c5 = True
                
            # Identify Cell 6 (Visual Auditor)
            if "# Chama o benchmark com 30 amostras" in content and "ref = json.loads" in content:
                # Replace the entire try-except block reading the JSON
                old_source = c['source']
                new_source = []
                skip = False
                for line in old_source:
                    if "try:" in line and "treino_ref.json" in "".join(old_source):
                        skip = True
                        new_source.append("DS = DATASET.lower().strip()\n")
                        new_source.append("MD = MODEL.lower().strip()\n")
                        continue
                    if skip:
                        if "DS =" in line and "MD =" in line and "except" not in line:
                            skip = False # Found the end of the block
                        continue
                    new_source.append(line)
                
                c['source'] = new_source
                updated_c6 = True

    if updated_c5 and updated_c6:
        with open('experimento_federado.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Módulos de Inteligência Arbitrária refatorados com sucesso! Kaggle Standard Aplicado!")
    else:
        print(f"Aviso: Célula 5 atualizada: {updated_c5} | Célula 6 atualizada: {updated_c6}")

except Exception as e:
    print(f"Erro: {e}")
