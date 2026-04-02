import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ResultsPage() {
  const [experiments, setExperiments] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch("/api/results/experiments")
      .then((res) => res.json())
      .then((payload) => {
        const items = payload.items || [];
        setExperiments(items);
        if (items.length > 0) {
          setSelectedId(items[0].id);
        }
      })
      .catch(() => setExperiments([]));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setMetrics(null);
      return;
    }

    fetch(`/api/results/experiments/${selectedId}`)
      .then((res) => res.json())
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, [selectedId]);

  const chartData = metrics?.history || [];

  return (
    <section className="grid gap-4">
      <div className="rounded-xl border border-line bg-panel p-4 font-mono">
        <h2 className="text-lg font-semibold text-neon">Resultados e Relatórios</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-[280px_1fr]">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Experimento</label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="mt-2 w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 text-sm"
            >
              {experiments.length === 0 ? <option value="">Sem experimentos</option> : null}
              {experiments.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id}
                </option>
              ))}
            </select>
          </div>
          <div className="rounded-md border border-line bg-[#0a111b] p-3 text-xs text-slate-300">
            Cada execução salva logs, métricas, tabelas e figuras na pasta central de resultados por ID de experimento.
          </div>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <MetricCard label="Dataset" value={metrics?.dataset || "-"} />
          <MetricCard label="Modelo" value={metrics?.model ?? "-"} />
          <MetricCard label="Distribuição" value={formatDistribution(metrics?.distribution)} />
          <MetricCard label="Clientes" value={metrics?.clients ?? "-"} />
          <MetricCard label="Loss Final" value={metrics?.final_loss ?? "-"} />
          <MetricCard label="Acurácia Final" value={metrics?.final_accuracy ?? "-"} />
          <MetricCard label="AWGN" value={formatAwgn(metrics?.awgn)} />
          <MetricCard label="Ruído" value={metrics?.noise ? JSON.stringify(metrics.noise) : "-"} />
        </div>
      </div>

      <div className="rounded-xl border border-line bg-panel p-4 font-mono">
        <h3 className="mb-3 text-sm uppercase tracking-wide text-slate-300">Curvas de Convergência</h3>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
              <XAxis dataKey="epoch" stroke="#9aa7bd" />
              <YAxis stroke="#9aa7bd" />
              <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d" }} />
              <Line type="monotone" dataKey="loss" stroke="#ffd166" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="accuracy" stroke="#00f6a2" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ArtifactImage title="Convergência da Loss" src={metrics?.figures?.loss} />
        <ArtifactImage title="Convergência da Acurácia" src={metrics?.figures?.accuracy} />
        <ArtifactImage title="Reconstrução de Imagem" src={metrics?.figures?.reconstruction} full />
      </div>

      <div className="rounded-xl border border-line bg-panel p-4 font-mono text-sm text-slate-300">
        <h3 className="text-sm uppercase tracking-wide text-slate-300">Tabelas Exportadas</h3>
        <div className="mt-2 flex flex-wrap gap-3">
          {metrics?.tables?.csv ? (
            <a className="rounded-md border border-line bg-[#0d1420] px-3 py-1.5" href={`/api${metrics.tables.csv}`} target="_blank" rel="noreferrer">
              resultados.csv
            </a>
          ) : null}
          {metrics?.tables?.tex ? (
            <a className="rounded-md border border-line bg-[#0d1420] px-3 py-1.5" href={`/api${metrics.tables.tex}`} target="_blank" rel="noreferrer">
              resultados.tex
            </a>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="rounded-md border border-line bg-[#0a111b] p-3">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-base font-semibold text-text">{String(value)}</p>
    </div>
  );
}

function ArtifactImage({ title, src, full = false }) {
  return (
    <div className={`rounded-xl border border-line bg-panel p-4 font-mono ${full ? "md:col-span-2" : ""}`}>
      <h4 className="mb-2 text-sm uppercase tracking-wide text-slate-300">{title}</h4>
      {src ? <img src={`/api${src}`} alt={title} className="w-full rounded-md border border-line" /> : <p className="text-sm text-slate-400">Sem figura disponível.</p>}
    </div>
  );
}

function formatDistribution(distribution) {
  if (!distribution || distribution === "iid") {
    return "IID";
  }
  if (distribution === "non_iid") {
    return "Não-IID";
  }
  return String(distribution);
}

function formatAwgn(awgn) {
  if (!awgn || !awgn.enabled) {
    return "Desligado";
  }
  if (awgn.snr_db === null || awgn.snr_db === undefined) {
    return "Ligado";
  }
  return `Ligado (${awgn.snr_db} dB)`;
}
