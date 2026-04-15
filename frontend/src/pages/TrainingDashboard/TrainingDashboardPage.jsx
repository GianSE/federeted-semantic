import { useEffect, useMemo, useState } from "react";
import TerminalLogWindow from "../../components/TerminalLogWindow";

export default function TrainingDashboardPage() {
  const [dataset, setDataset] = useState("mnist");
  const [model, setModel] = useState("cnn_vae");
  const [distribution, setDistribution] = useState("iid");
  const [awgn, setAwgn] = useState({ enabled: false, snr_db: 10 });
  const [clients, setClients] = useState(2);
  const [epochs, setEpochs] = useState(3);
  const [realTraining, setRealTraining] = useState(false);
  const [noise, setNoise] = useState({
    channel: 0,
    packet_loss: 0,
    latency: 0,
    client_drift: 0,
  });
  const [logsByTarget, setLogsByTarget] = useState({ server: [] });
  const [activeTarget, setActiveTarget] = useState("server");
  const [connected, setConnected] = useState(false);
  const [streamEnabled, setStreamEnabled] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isRealMode, setIsRealMode] = useState(false); // reflects server state
  const [isPaused, setIsPaused] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [activeTab, setActiveTab] = useState("topology");

  const logTargets = useMemo(() => {
    const out = ["server"];
    for (let i = 1; i <= clients; i += 1) {
      out.push(`client-${i}`);
    }
    return out;
  }, [clients]);

  const streamUrl = useMemo(() => `/api/logs/stream?target=${activeTarget}`, [activeTarget]);

  useEffect(() => {
    if (!streamEnabled) {
      setConnected(false);
      return;
    }

    const source = new EventSource(streamUrl);

    source.onopen = () => {
      setConnected(true);
      setLogsByTarget((prev) => ({ ...prev, [activeTarget]: [] }));
    };
    source.onerror = () => setConnected(false);
    source.onmessage = (event) => {
      if (!event.data) return;
      if (event.data.startsWith("[heartbeat]")) return;

      setLogsByTarget((prev) => {
        const current = prev[activeTarget] || [];
        return { ...prev, [activeTarget]: [...current.slice(-180), event.data] };
      });

      if (event.data.includes("[done]") || event.data.includes("[stopped]")) {
        setIsTraining(false);
        setIsPaused(false);
      }
    };

    return () => source.close();
  }, [streamEnabled, streamUrl, activeTarget]);

  useEffect(() => {
    if (!logTargets.includes(activeTarget)) {
      setActiveTarget("server");
    }
  }, [activeTarget, logTargets]);

  useEffect(() => {
    fetch("/api/training/status")
      .then((res) => (res.ok ? res.json() : null))
      .then((status) => {
        if (!status) return;
        setIsTraining(Boolean(status.running));
        setIsPaused(Boolean(status.paused));
        setIsRealMode(Boolean(status.real_training));
        if (status.running) setStreamEnabled(true);
      })
      .catch(() => {});
  }, []);

  async function startTraining() {
    setActionPending(true);
    const response = await fetch("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        model,
        distribution,
        clients,
        noise,
        awgn,
        real_training: realTraining,
        epochs,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      setLogsByTarget((prev) => {
        const current = prev.server || [];
        return { ...prev, server: [...current, `[error] falha ao iniciar: ${text}`] };
      });
      setActionPending(false);
      return;
    }

    const payload = await response.json();
    if (payload.status === "already_running") {
      setIsTraining(true);
      setStreamEnabled(true);
      setActionPending(false);
      return;
    }

    setStreamEnabled(true);
    setIsTraining(true);
    setIsRealMode(realTraining);
    setIsPaused(false);

    const modeLabel = realTraining ? `REAL (PyTorch) épocas=${epochs}` : "SIMULAÇÃO";
    setLogsByTarget((prev) => {
      const current = prev.server || [];
      return {
        ...prev,
        server: [
          ...current,
          `[controle] treino iniciado | modo=${modeLabel} | dataset=${dataset} | modelo=${model} | clients=${clients}`,
        ],
      };
    });
    setActionPending(false);
  }

  async function stopTraining() {
    setActionPending(true);
    const response = await fetch("/api/training/stop", { method: "POST", headers: { "Content-Type": "application/json" } });
    if (response.ok) {
      setLogsByTarget((prev) => {
        const current = prev.server || [];
        return { ...prev, server: [...current, "[controle] parada solicitada"] };
      });
    }
    setActionPending(false);
  }

  async function togglePause() {
    setActionPending(true);
    const endpoint = isPaused ? "/api/training/resume" : "/api/training/pause";
    const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" } });

    if (response.ok) {
      setIsPaused((prev) => !prev);
      setLogsByTarget((prev) => {
        const current = prev.server || [];
        return { ...prev, server: [...current, isPaused ? "[controle] treino retomado" : "[controle] treino pausado"] };
      });
    }
    setActionPending(false);
  }

  async function clearLogs() {
    setActionPending(true);
    await fetch("/api/training/logs/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clients }),
    });

    const empty = { server: [] };
    for (let i = 1; i <= clients; i += 1) {
      empty[`client-${i}`] = [];
    }
    setLogsByTarget(empty);
    setActionPending(false);
  }

  function onNoiseChange(key, value) {
    setNoise((prev) => ({ ...prev, [key]: Number(value) }));
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[400px_1fr]">
      {/* ── Left: Controls ───────────────────────────────────────────── */}
      <div className="flex flex-col gap-4">

        {/* ── MODE SELECTOR ──────────────────────────────────────────── */}
        <div className={`rounded-xl border p-4 font-mono ${
          realTraining
            ? "border-orange-500 bg-[#1a0e00]"
            : "border-line bg-panel"
        }`}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className={`text-sm font-bold ${ realTraining ? "text-orange-400" : "text-slate-300" }`}>
                {realTraining ? "🔥 Modo Real (PyTorch)" : "🎭 Modo Simulação"}
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                {realTraining
                  ? "Gradientes reais · Pesos salvos em .pth · Uso nas páginas /semantic e /benchmark"
                  : "Demo rápida · Curvas FedAvg sintéticas · Não gera pesos de modelo"}
              </p>
            </div>
            <button
              id="mode-toggle-btn"
              type="button"
              onClick={() => setRealTraining((p) => !p)}
              disabled={isTraining}
              className={`rounded-md px-3 py-1.5 text-xs font-bold uppercase tracking-wider border transition disabled:opacity-40 ${
                realTraining
                  ? "bg-orange-900 border-orange-500 text-orange-300 hover:bg-orange-800"
                  : "bg-[#0d1420] border-line text-slate-400 hover:text-slate-200"
              }`}
            >
              {realTraining ? "Real ON" : "Simulação"}
            </button>
          </div>

          {realTraining && (
            <div className="mt-2 pt-3 border-t border-orange-900">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-slate-400">Épocas por cliente</span>
                <span className="text-orange-400 font-bold">{epochs}</span>
              </div>
              <input
                type="range" min="1" max="15" value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-orange-400 disabled:opacity-50"
              />
              <p className="text-[10px] text-slate-500 mt-1">
                {epochs} época{epochs > 1 ? "s" : ""} × {clients} cliente{clients > 1 ? "s" : ""}
                {" = ~"}{Math.round(epochs * clients * (dataset === "cifar10" ? 4 : 2))} min (CPU)
              </p>
              <div className="mt-2 rounded bg-[#2a1200] border border-orange-900 p-2 text-[10px] text-orange-300">
                ⚠ O treino real pode demorar alguns minutos por cliente (CPU).
                Acompanhe os logs no terminal ao lado.
              </div>
            </div>
          )}
        </div>

        {/* ── Main config panel ──────────────────────────────────────── */}
        <div className="rounded-xl border border-line bg-panel p-6 font-mono shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-neon opacity-5 blur-3xl pointer-events-none"></div>
          
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-neon">Controle Federado</h2>
            {isTraining && (
              <span className={`text-[10px] font-bold uppercase px-2 py-1 rounded border ${
                isRealMode
                  ? "border-orange-500 bg-[#1a0e00] text-orange-400"
                  : "border-neon bg-[#073529] text-neon"
              }`}>
                {isRealMode ? "🔥 Real" : "🎭 Sim"}
              </span>
            )}
          </div>

          {/* Abas */}
          <div className="flex mb-6 border-b border-[#121c2e]">
            <button className={`flex-1 pb-2 text-xs font-bold uppercase tracking-wider transition ${activeTab === 'topology' ? 'border-b-2 border-neon text-neon' : 'text-slate-500'}`} onClick={() => setActiveTab('topology')}>Topologia</button>
            <button className={`flex-1 pb-2 text-xs font-bold uppercase tracking-wider transition ${activeTab === 'genai' ? 'border-b-2 border-neon text-neon' : 'text-slate-500'}`} onClick={() => setActiveTab('genai')}>GenAI</button>
            <button className={`flex-1 pb-2 text-xs font-bold uppercase tracking-wider transition ${activeTab === 'channel' ? 'border-b-2 border-neon text-neon' : 'text-slate-500'}`} onClick={() => setActiveTab('channel')}>Canal (Ruído)</button>
          </div>

          {/* Conteúdo Aba 1: Topologia */}
          {activeTab === 'topology' && (
            <div className="animate-fade-in space-y-5">
              <div>
                <label className="text-xs uppercase tracking-wide text-slate-400">Distribuição Hashed</label>
                <select value={distribution} onChange={(e) => setDistribution(e.target.value)} disabled={isTraining} className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-50 transition-colors focus:border-neon focus:outline-none">
                  <option value="iid">IID (Independente e Idêntica)</option>
                  <option value="non_iid">Não-IID (Caótica/Amostrada)</option>
                </select>
              </div>
              <div>
                <div className="flex justify-between items-center">
                  <label className="text-xs uppercase tracking-wide text-slate-400">Total Edge Clients</label>
                  <span className="text-neon font-bold text-lg">{clients}</span>
                </div>
                <input type="range" min="1" max="10" value={clients} onChange={(e) => setClients(Number(e.target.value))} disabled={isTraining} className="mt-2 w-full disabled:opacity-50 accent-neon" />
                <p className="text-[10px] text-slate-500 mt-1">Aumentar causa maior overhead no servidor coordenador e gargalos.</p>
              </div>
            </div>
          )}

          {/* Conteúdo Aba 2: GenAI */}
          {activeTab === 'genai' && (
            <div className="animate-fade-in space-y-5">
              <div>
                <label className="text-xs uppercase tracking-wide text-slate-400">Dataset de Destino</label>
                <select value={dataset} onChange={(e) => setDataset(e.target.value)} disabled={isTraining} className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-50 focus:border-neon focus:outline-none">
                  <option value="fashion">Fashion-MNIST</option>
                  <option value="mnist">MNIST</option>
                  <option value="cifar10">CIFAR-10 (Colorido)</option>
                </select>
              </div>
              <div>
                <label className="text-xs uppercase tracking-wide text-slate-400">Backbone da IA</label>
                <select value={model} onChange={(e) => setModel(e.target.value)} disabled={isTraining} className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-50 focus:border-neon focus:outline-none">
                  <option value="cnn_vae">GenAI Variational AE (Recomendado)</option>
                  <option value="cnn_ae">CNN Autoencoder Direto</option>
                  <option value="ae">MLP Linear Clássico</option>
                </select>
              </div>
            </div>
          )}

          {/* Conteúdo Aba 3: Canal (Ruído) */}
          {activeTab === 'channel' && (
            <div className="animate-fade-in space-y-4">
              <div className="rounded-md border border-line bg-[#0a111b] p-3 text-sm mb-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wide text-slate-400 font-bold">Simulador AWGN</span>
                  <button type="button" onClick={() => setAwgn((prev) => ({ ...prev, enabled: !prev.enabled }))} disabled={isTraining} className={`rounded uppercase text-[10px] px-2 py-1 font-bold transition disabled:opacity-50 ${awgn.enabled ? "bg-[#073529] text-neon border border-neon" : "bg-[#1f2937] text-slate-400 border border-transparent"}`}>
                    {awgn.enabled ? "Ativo" : "Inativo"}
                  </button>
                </div>
                {awgn.enabled && (
                  <div className="mt-4">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-slate-400">SNR Base (dB)</span>
                      <span className="text-neon">{awgn.snr_db}</span>
                    </div>
                    <input type="range" min="0" max="30" value={awgn.snr_db} onChange={(e) => setAwgn((prev) => ({ ...prev, snr_db: Number(e.target.value) }))} disabled={isTraining} className="w-full accent-neon disabled:opacity-50"/>
                  </div>
                )}
              </div>
              
              <NoiseControl label="Ruído de Fundo (White)" value={noise.channel} max={100} onChange={(v) => onNoiseChange("channel", v)} disabled={isTraining} unit="%" />
              <NoiseControl label="Drop de Pacotes Semânticos" value={noise.packet_loss} max={50} onChange={(v) => onNoiseChange("packet_loss", v)} disabled={isTraining} unit="%" />
              <NoiseControl label="Desvio (Client Drift)" value={noise.client_drift} max={80} onChange={(v) => onNoiseChange("client_drift", v)} disabled={isTraining} unit="var" />
            </div>
          )}
        </div>

        {/* ── Status + Action Bar ─────────────────────────────────── */}
        <div className="rounded-xl border border-line bg-panel p-4 flex flex-col justify-between font-mono">
          <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400 mb-4 px-2">
            <span>Link do Log:</span>
            <span className={connected ? "text-neon font-bold flex items-center gap-1" : "text-warn font-bold flex items-center gap-1"}>
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-neon animate-pulse' : 'bg-warn'}`}></span>
              {connected ? "Conectado" : "Offline"} ({activeTarget})
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <button
              id="start-stop-btn"
              onClick={isTraining ? stopTraining : startTraining}
              disabled={actionPending}
              className={`w-full rounded-md border px-4 py-3 text-sm font-bold uppercase tracking-wider transition-all duration-200 shadow-lg ${
                actionPending
                  ? "scale-[0.98] animate-pulse border-warn bg-[#3d3313] text-warn shadow-none"
                  : isTraining
                  ? "border-[#ff7b7b] bg-[#3b1a1a] text-[#ff9a9a] hover:bg-[#522929]"
                  : realTraining
                  ? "border-orange-500 bg-[#1a0e00] text-orange-300 hover:bg-[#261200]"
                  : "border-neon bg-[#073529] text-neon hover:bg-[#0b4a3a]"
              }`}
            >
              {isTraining
                ? "PARAR TREINAMENTO"
                : realTraining
                ? "🔥 INICIAR TREINO REAL (PyTorch)"
                : "▶ INICIAR SIMULAÇÃO FEDAVG"}
            </button>
            <button
              onClick={togglePause}
              disabled={!isTraining || actionPending}
              className="w-full rounded-md border border-line bg-[#0d1420] px-4 py-2 text-xs font-bold uppercase text-slate-300 disabled:opacity-40 transition-colors hover:bg-[#1a2536]"
            >
              {isPaused ? "Retomar Execução" : "Pausar Orquestrador"}
            </button>
          </div>
        </div>
      </div>

      {/* Direita: Terminal */}
      <div className="rounded-xl border border-line bg-panel flex flex-col overflow-hidden">
        <div className="bg-[#0b1220] border-b border-line p-3 flex flex-wrap gap-2 items-center justify-between">
          <div className="flex gap-2 flex-wrap">
            {logTargets.map((target) => (
              <button key={target} onClick={() => setActiveTarget(target)} className={`rounded-md border px-3 py-1 font-mono text-xs uppercase transition-colors ${activeTarget === target ? "border-neon bg-[#0b2a22] text-neon" : "border-line bg-transparent text-slate-400 hover:text-slate-200"}`}>
                {target}
              </button>
            ))}
          </div>
          <button onClick={clearLogs} disabled={actionPending} className="rounded border border-line bg-[#151e2e] px-3 py-1 text-xs text-slate-400 hover:text-white transition">
            Limpar Console
          </button>
        </div>
        
        <div className="flex-1 bg-black">
          <TerminalLogWindow logs={logsByTarget[activeTarget] || []} title={`/> tail -f ${activeTarget}.log`} streamStarted={streamEnabled} />
        </div>
      </div>
    </section>
  );
}

function NoiseControl({ label, value, max, onChange, disabled = false, unit = "" }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1 font-mono">
        <span className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-400">{label}</span>
        <span className="text-neon text-xs">{value} {unit}</span>
      </div>
      <input type="range" min="0" max={max} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} className="w-full disabled:opacity-50 accent-[#ffd166]" />
    </div>
  );
}
