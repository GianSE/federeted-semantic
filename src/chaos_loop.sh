#!/bin/sh

# Instala dependências
apk add --no-cache iproute2

CONFIG_FILE="/app/chaos_config.txt"
LAST_CFG=""

echo "😈 Chaos Loop Iniciado..."

# Limpa regras ao iniciar
tc qdisc del dev eth0 root 2>/dev/null

while true; do
    if [ -f "$CONFIG_FILE" ]; then
        # Formato esperado: ON perda delay corrupcao duplicacao
        # Ex: ON 0.1 500 0.5 1.0
        CURRENT_CFG=$(cat "$CONFIG_FILE")
        
        if [ "$CURRENT_CFG" != "$LAST_CFG" ]; then
            echo "⚡ Mudança: $CURRENT_CFG"
            
            # Remove regras antigas
            tc qdisc del dev eth0 root 2>/dev/null
            
            STATUS=$(echo $CURRENT_CFG | cut -d' ' -f1)
            LOSS=$(echo $CURRENT_CFG | cut -d' ' -f2)
            DELAY=$(echo $CURRENT_CFG | cut -d' ' -f3)
            CORRUPT=$(echo $CURRENT_CFG | cut -d' ' -f4)
            DUPLICATE=$(echo $CURRENT_CFG | cut -d' ' -f5) # <--- O 5º ELEMENTO
            
            if [ "$STATUS" = "ON" ]; then
                echo "🔥 Aplicando: Loss $LOSS% | Delay ${DELAY}ms | Corrupt $CORRUPT% | Duplicate $DUPLICATE%"
                # Comando tc completo com todas as maldades
                tc qdisc add dev eth0 root netem loss $LOSS% delay ${DELAY}ms corrupt $CORRUPT% duplicate $DUPLICATE%
            else
                echo "🕊️ Caos Desativado"
            fi
            
            LAST_CFG="$CURRENT_CFG"
        fi
    fi
    sleep 2
done