import subprocess
import os

def compilar_artigo(arquivo_principal="main.tex"):
    # Pega apenas o nome sem a extensão (ex: "main")
    nome_base = os.path.splitext(arquivo_principal)[0]

    # Verifica se o arquivo .tex realmente existe na pasta
    if not os.path.exists(arquivo_principal):
        print(f"Erro: O arquivo '{arquivo_principal}' não foi encontrado neste diretório.")
        return False

    # Lista de comandos que precisam ser executados em ordem
    comandos = [
        ["pdflatex", "-interaction=nonstopmode", arquivo_principal],
        ["bibtex", nome_base],
        ["pdflatex", "-interaction=nonstopmode", arquivo_principal],
        ["pdflatex", "-interaction=nonstopmode", arquivo_principal]
    ]

    print(f"Iniciando compilação do arquivo: {arquivo_principal}\n" + "-"*40)

    for i, cmd in enumerate(comandos, start=1):
        comando_str = " ".join(cmd)
        print(f"[{i}/{len(comandos)}] Executando: {comando_str}")
        
        try:
            # Executa o comando
            resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Verifica se deu erro (código de retorno diferente de 0)
            if resultado.returncode != 0:
                print(f"\n[!] Ocorreu um erro no passo {i}.")
                print("Últimas linhas da saída de erro:")
                # Mostra as últimas linhas do log para ajudar a identificar o erro
                linhas_saida = resultado.stdout.split('\n')
                print('\n'.join(linhas_saida[-20:])) 
                return False
                
            print("OK.")
            
        except FileNotFoundError:
            print(f"\n[!] Erro crítico: O comando '{cmd[0]}' não foi encontrado.")
            print("Certifique-se de que o MiKTeX, TeX Live ou MacTeX está instalado e configurado nas variáveis de ambiente (PATH).")
            return False

    print("-" * 40)
    print(f"Sucesso! O artigo foi compilado e o arquivo '{nome_base}.pdf' foi atualizado.")
    return True

if __name__ == "__main__":
    compilar_artigo("main.tex")