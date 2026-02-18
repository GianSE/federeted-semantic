#!/bin/sh

# Instala dependências de rede
apk add --no-cache iproute2

CONFIG_FILE="/app/chaos_config.txt"
LAST_CFG=""

echo "😈 Chaos Loop Iniciado. Aguardando comandos..."

# Garante que a rede comece limpa
tc qdisc del dev eth0 root 2>/dev/null

while true; do
    if [ -f "$CONFIG_FILE" ]; then
        # Lê a configuração (Formato esperado: ON/OFF loss delay)
        # Exemplo: ON 10% 200ms
        CURRENT_CFG=$(cat "$CONFIG_FILE")
        
        # Só aplica se a configuração mudou
        if [ "$CURRENT_CFG" != "$LAST_CFG" ]; then
            echo "⚡ Mudança detectada: $CURRENT_CFG"
            
            # Limpa regras antigas
            tc qdisc del dev eth0 root 2>/dev/null
            
            STATUS=$(echo $CURRENT_CFG | cut -d' ' -f1)
            LOSS=$(echo $CURRENT_CFG | cut -d' ' -f2)
            DELAY=$(echo $CURRENT_CFG | cut -d' ' -f3)
            
            if [ "$STATUS" = "ON" ]; then
                echo "🔥 Aplicando: Loss $LOSS | Delay $DELAY"
                tc qdisc add dev eth0 root netem loss $LOSS delay $DELAY
            else
                echo "🕊️ Caos Desativado (Rede Normal)"
            fi
            
            LAST_CFG="$CURRENT_CFG"
        fi
    fi
    sleep 2
done