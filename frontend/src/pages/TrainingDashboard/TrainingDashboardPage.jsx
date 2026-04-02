import { useEffect, useMemo, useState } from "react";
import TerminalLogWindow from "../../components/TerminalLogWindow";

export default function TrainingDashboardPage() {
  const [dataset, setDataset] = useState("mnist");
  const [model, setModel] = useState("ae");
  const [distribution, setDistribution] = useState("iid");
  const [awgn, setAwgn] = useState({ enabled: false, snr_db: 10 });
  const [clients, setClients] = useState(3);
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
  const [isPaused, setIsPaused] = useState(false);
  const [actionPending, setActionPending] = useState(false);

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

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (event) => {
      if (!event.data) {
        return;
      }

      if (event.data.startsWith("[heartbeat]")) {
        return;
      }

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
        if (!status) {
          return;
        }
        setIsTraining(Boolean(status.running));
        setIsPaused(Boolean(status.paused));
        if (status.running) {
          setStreamEnabled(true);
        }
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
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      setLogsByTarget((prev) => {
        const current = prev.server || [];
        return { ...prev, server: [...current, `[error] failed to start training: ${text}`] };
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
    setIsPaused(false);

    setLogsByTarget((prev) => {
      const current = prev.server || [];
      return {
        ...prev,
        server: [
          ...current,
          `[controle] treino iniciado dataset=${dataset} modelo=${model} distribution=${distribution} clients=${clients} awgn=${awgn.enabled ? `on:${awgn.snr_db}dB` : "off"}`,
        ],
      };
    });
    setActionPending(false);
  }

  async function stopTraining() {
    setActionPending(true);
    const response = await fetch("/api/training/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

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
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (response.ok) {
      setIsPaused((prev) => !prev);
      setLogsByTarget((prev) => {
        const current = prev.server || [];
        return {
          ...prev,
          server: [...current, isPaused ? "[controle] treino retomado" : "[controle] treino pausado"],
        };
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
    <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <div className="rounded-xl border border-line bg-panel p-4 font-mono">
        <h2 className="text-lg font-semibold text-neon">Treinamento Federado</h2>

        <label className="mt-4 block text-sm text-slate-300">Dataset</label>
        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          disabled={isTraining}
          className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-60"
        >
          <option value="mnist">MNIST</option>
          <option value="cifar10">CIFAR-10</option>
          <option value="cifar100">CIFAR-100</option>
        </select>

        <label className="mt-3 block text-sm text-slate-300">Modelo</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={isTraining}
          className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-60"
        >
          <option value="ae">AE</option>
          <option value="cnn_ae">CNN AE</option>
          <option value="cnn_vae">CNN VAE</option>
        </select>

        <label className="mt-3 block text-sm text-slate-300">Distribuição</label>
        <select
          value={distribution}
          onChange={(e) => setDistribution(e.target.value)}
          disabled={isTraining}
          className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm disabled:opacity-60"
        >
          <option value="iid">IID</option>
          <option value="non_iid">Não-IID</option>
        </select>

        <label className="mt-3 block text-sm text-slate-300">Clientes: {clients}</label>
        <input
          type="range"
          min="1"
          max="6"
          value={clients}
          onChange={(e) => setClients(Number(e.target.value))}
          disabled={isTraining}
          className="mt-2 w-full disabled:opacity-60"
        />

        <p className="mt-4 text-xs uppercase tracking-wide text-slate-400">Controladores de Ruído</p>
        <NoiseControl label="Ruído de Canal" value={noise.channel} max={100} onChange={(v) => onNoiseChange("channel", v)} disabled={isTraining} />
        <NoiseControl label="Perda de Pacotes" value={noise.packet_loss} max={30} onChange={(v) => onNoiseChange("packet_loss", v)} disabled={isTraining} />
        <NoiseControl label="Latência (ms)" value={noise.latency} max={400} onChange={(v) => onNoiseChange("latency", v)} disabled={isTraining} />
        <NoiseControl label="Drift de Cliente" value={noise.client_drift} max={40} onChange={(v) => onNoiseChange("client_drift", v)} disabled={isTraining} />

        <div className="mt-4 rounded-md border border-line bg-[#0a111b] p-3">
          <p className="text-xs uppercase tracking-wide text-slate-400">AWGN</p>
          <button
            type="button"
            onClick={() => setAwgn((prev) => ({ ...prev, enabled: !prev.enabled }))}
            disabled={isTraining}
            className={`mt-2 w-full rounded-md border px-3 py-2 text-sm font-semibold disabled:opacity-60 ${
              awgn.enabled ? "border-neon bg-[#073529] text-neon" : "border-line bg-[#0d1420] text-slate-300"
            }`}
          >
            {awgn.enabled ? "AWGN ligado" : "AWGN desligado"}
          </button>

          {awgn.enabled ? (
            <div className="mt-3">
              <label className="block text-xs text-slate-300">SNR (dB): {awgn.snr_db}</label>
              <input
                type="range"
                min="0"
                max="30"
                value={awgn.snr_db}
                onChange={(e) => setAwgn((prev) => ({ ...prev, snr_db: Number(e.target.value) }))}
                disabled={isTraining}
                className="w-full disabled:opacity-60"
              />
            </div>
          ) : null}
        </div>

        <button
          onClick={isTraining ? stopTraining : startTraining}
          disabled={actionPending}
          className={`mt-4 w-full rounded-md border px-3 py-2 text-sm font-semibold transition-transform duration-150 ${
            actionPending
              ? "scale-95 animate-pulse border-warn bg-[#3d3313] text-warn"
              : isTraining
                ? "border-[#ff7b7b] bg-[#3b1a1a] text-[#ff9a9a]"
                : "border-neon bg-[#073529] text-neon"
          }`}
        >
          {isTraining ? "Parar Treinamento" : "Iniciar Treinamento"}
        </button>

        <div className="mt-2">
          <button
            onClick={togglePause}
            disabled={!isTraining || actionPending}
            className="w-full rounded-md border border-line bg-[#0d1420] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50"
          >
            {isPaused ? "Retomar" : "Pausar"}
          </button>
        </div>

        <div className="mt-4 rounded-md border border-line bg-[#0a111b] p-3 text-xs text-slate-300">
          Status do stream: <span className={connected ? "text-neon" : "text-warn"}>{connected ? "conectado" : "desconectado"}</span> ({activeTarget})
        </div>
      </div>

      <div className="rounded-xl border border-line bg-panel p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          {logTargets.map((target) => (
            <button
              key={target}
              onClick={() => setActiveTarget(target)}
              className={`rounded-md border px-3 py-1.5 font-mono text-xs ${
                activeTarget === target ? "border-neon bg-[#0b2a22] text-neon" : "border-line bg-[#0d1420] text-slate-300"
              }`}
            >
              {target}
            </button>
          ))}
        </div>

        <TerminalLogWindow logs={logsByTarget[activeTarget] || []} title={`${activeTarget}.log`} streamStarted={streamEnabled} />

        <button
          onClick={clearLogs}
          disabled={actionPending}
          className="mt-3 w-full rounded-md border border-line bg-[#0d1420] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50"
        >
          Limpar Logs do Terminal
        </button>
      </div>
    </section>
  );
}

function NoiseControl({ label, value, max, onChange, disabled = false }) {
  return (
    <div className="mt-2">
      <label className="block text-xs text-slate-300">
        {label}: {value}
      </label>
      <input
        type="range"
        min="0"
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full disabled:opacity-60"
      />
    </div>
  );
}
