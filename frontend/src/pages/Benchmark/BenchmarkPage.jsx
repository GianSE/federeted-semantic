import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

// ─── constants ────────────────────────────────────────────────────────────────

const DATASET_COLORS = {
  mnist:   "#00f6a2",
  fashion: "#ffd166",
  cifar10: "#489dff",
};

const MODEL_COLORS = {
  cnn_vae: "#00f6a2",
  cnn_ae:  "#ff7b7b",
};

const BITS_LABELS = { 4: "Int4", 8: "Int8", 16: "Int16", 32: "Float32" };

const SCALABILITY_DEVICES = [1, 5, 10, 50, 100];

// ─── helpers ──────────────────────────────────────────────────────────────────

/**
 * Format bytes into human-readable KB / MB.
 * @param {number} bytes
 * @returns {string}
 */
function fmtBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  if (bytes >= 1024)        return `${(bytes / 1024).toFixed(2)} KB`;
  return `${bytes} B`;
}

/** Round to N decimal places */
const r = (v, n = 3) => (typeof v === "number" ? v.toFixed(n) : "—");

// ─── sub-components ───────────────────────────────────────────────────────────

function StatBadge({ label, value, unit = "", color = "text-neon" }) {
  return (
    <div className="rounded-md border border-line bg-[#0a111b] p-3 flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-mono">{label}</span>
      <span className={`text-xl font-bold font-mono ${color}`}>
        {value}
        {unit && <span className="text-sm ml-1 text-slate-400">{unit}</span>}
      </span>
    </div>
  );
}

function ScalabilityTable({ results }) {
  // Pick the best result per dataset (highest compression ratio, cnn_vae preferred)
  const byDataset = {};
  for (const r of results) {
    if (r.status !== "ok") continue;
    if (!byDataset[r.dataset] || r.model === "cnn_vae") {
      byDataset[r.dataset] = r;
    }
  }

  const entries = Object.values(byDataset);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-xl border border-line bg-panel p-6">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-1">
        Simulação de Escalabilidade — Banda Necessária por N Dispositivos
      </h3>
      <p className="text-xs text-slate-400 font-mono mb-4">
        Comparativo de tráfego bruto vs. semântico (latente) ao escalar para N dispositivos simultâneos.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="border-b border-line text-slate-400 uppercase tracking-wide">
              <th className="text-left py-2 pr-4">Dataset</th>
              <th className="text-right py-2 pr-4">Razão</th>
              {SCALABILITY_DEVICES.map((n) => (
                <th key={n} className="text-right py-2 pr-4" colSpan={2}>
                  {n} dispositivo{n > 1 ? "s" : ""}
                </th>
              ))}
            </tr>
            <tr className="border-b border-line text-slate-500">
              <th className="text-left py-1"></th>
              <th className="text-right py-1 pr-4"></th>
              {SCALABILITY_DEVICES.map((n) => (
                <>
                  <th key={`${n}-raw`} className="text-right py-1 pr-2 text-red-400">Bruto</th>
                  <th key={`${n}-lat`} className="text-right py-1 pr-4 text-neon">Latente</th>
                </>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => {
              const scalability = entry.scalability || {};
              return (
                <tr key={entry.dataset} className="border-b border-[#121c2e] hover:bg-[#0a111b] transition">
                  <td className="py-2 pr-4 font-bold" style={{ color: DATASET_COLORS[entry.dataset] }}>
                    {entry.dataset.toUpperCase()}
                  </td>
                  <td className="text-right pr-4 text-neon font-bold">
                    {r(entry.compression_ratio_mean, 1)}×
                  </td>
                  {SCALABILITY_DEVICES.map((n) => {
                    const key = `${n}_devices`;
                    const s = scalability[key] || {};
                    return (
                      <>
                        <td key={`${n}-raw`} className="text-right pr-2 text-red-400">
                          {s.total_original_kb != null ? `${s.total_original_kb} KB` : "—"}
                        </td>
                        <td key={`${n}-lat`} className="text-right pr-4 text-neon">
                          {s.total_latent_kb != null ? `${s.total_latent_kb} KB` : "—"}
                        </td>
                      </>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

/**
 * BenchmarkPage
 *
 * Runs a cross-dataset benchmark via POST /api/experiment/benchmark and
 * presents aggregate quality metrics (MSE, PSNR, SSIM, compression ratio)
 * in a structured table plus bar charts.
 *
 * This is the primary scientific evidence page for the research hypothesis:
 * "Latent representations reduce bandwidth while preserving semantic information."
 */
export default function BenchmarkPage() {
  const [bits, setBits]             = useState(8);
  const [numSamples, setNumSamples] = useState(20);
  const [loading, setLoading]       = useState(false);
  const [data, setData]             = useState(null);
  const [error, setError]           = useState(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await fetch("/api/experiment/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasets: ["mnist", "fashion", "cifar10"],
          models: ["cnn_vae", "cnn_ae"],
          bits,
          num_samples: numSamples,
          seed: 42,
        }),
      });

      const json = await response.json();
      if (!response.ok) throw new Error(json.error || "Erro desconhecido");
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const okResults = data?.results?.filter((r) => r.status === "ok") ?? [];

  // ── chart data ─────────────────────────────────────────────────────────────
  const compressionBarData = okResults.map((r) => ({
    name: `${r.dataset.toUpperCase()}\n(${r.model})`,
    ratio: parseFloat(r.compression_ratio_mean?.toFixed(1) ?? 0),
    fill: DATASET_COLORS[r.dataset],
  }));

  const qualityBarData = okResults.map((r) => ({
    name: `${r.dataset.toUpperCase()} / ${r.model === "cnn_vae" ? "VAE" : "AE"}`,
    PSNR: parseFloat(r.psnr_mean?.toFixed(1) ?? 0),
    SSIM: parseFloat((r.ssim_mean * 100)?.toFixed(1) ?? 0),
    fill: DATASET_COLORS[r.dataset],
  }));

  const radarData = [
    { metric: "SSIM×100" },
    { metric: "PSNR÷3" },
    { metric: "CR×3" },
    { metric: "-MSE×1k" },
  ];
  // Populate radar data for the first two results (VAE vs AE on same dataset)
  const vaeResult = okResults.find((r) => r.model === "cnn_vae");
  const aeResult  = okResults.find((r) => r.model === "cnn_ae" && r.dataset === vaeResult?.dataset);
  if (vaeResult) {
    radarData[0].VAE = parseFloat(((vaeResult.ssim_mean ?? 0) * 100).toFixed(1));
    radarData[1].VAE = parseFloat(((vaeResult.psnr_mean ?? 0) / 3).toFixed(1));
    radarData[2].VAE = parseFloat(((vaeResult.compression_ratio_mean ?? 0) * 3).toFixed(1));
    radarData[3].VAE = parseFloat((-((vaeResult.mse_mean ?? 0) * 1000)).toFixed(1));
  }
  if (aeResult) {
    radarData[0].AE = parseFloat(((aeResult.ssim_mean ?? 0) * 100).toFixed(1));
    radarData[1].AE = parseFloat(((aeResult.psnr_mean ?? 0) / 3).toFixed(1));
    radarData[2].AE = parseFloat(((aeResult.compression_ratio_mean ?? 0) * 3).toFixed(1));
    radarData[3].AE = parseFloat((-((aeResult.mse_mean ?? 0) * 1000)).toFixed(1));
  }

  return (
    <div className="grid gap-6">
      {/* ── Header & Controls ────────────────────────────────────────────── */}
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-1">
          Benchmark Científico — Comparação Multi-Dataset
        </h2>
        <p className="text-sm text-slate-400 mb-6 font-mono">
          Avalia MSE, PSNR, SSIM e razão de compressão para MNIST, Fashion-MNIST e CIFAR-10.
          Esta é a evidência central da hipótese de pesquisa.
        </p>

        <div className="flex flex-wrap gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 min-w-[160px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">
              Quantização (bits)
            </label>
            <select
              value={bits}
              onChange={(e) => setBits(Number(e.target.value))}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              {[4, 8, 16, 32].map((b) => (
                <option key={b} value={b}>
                  {BITS_LABELS[b]} ({b} bits)
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 min-w-[160px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">
              Amostras por combinação
            </label>
            <select
              value={numSamples}
              onChange={(e) => setNumSamples(Number(e.target.value))}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              {[10, 20, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n} imagens
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          id="benchmark-run-btn"
          onClick={handleRun}
          disabled={loading}
          className="rounded-md bg-[#073529] border border-neon text-neon px-6 py-2 font-mono text-sm hover:bg-[#0b2a22] transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Rodando benchmark...
            </span>
          ) : (
            "▶ Executar Benchmark Completo"
          )}
        </button>

        {error && (
          <div className="mt-4 rounded-md border border-red-800 bg-[#1a0f0f] p-3 text-sm text-red-400 font-mono">
            Erro: {error}
          </div>
        )}

        {data && !loading && (
          <div className="mt-4 flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="w-2 h-2 rounded-full bg-neon"></span>
            Benchmark concluído em {new Date(data.timestamp * 1000).toLocaleTimeString()} —{" "}
            {data.results?.length} combinações avaliadas
          </div>
        )}
      </div>

      {/* ── Results Table ─────────────────────────────────────────────────── */}
      {okResults.length > 0 && (
        <div className="rounded-xl border border-line bg-panel p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
            Tabela de Resultados — Evidência da Hipótese
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-line text-slate-400 uppercase tracking-wider text-[10px]">
                  <th className="text-left py-3 pr-4">Dataset</th>
                  <th className="text-left py-3 pr-4">Modelo</th>
                  <th className="text-left py-3 pr-4">Pesos</th>
                  <th className="text-right py-3 pr-4">MSE ↓</th>
                  <th className="text-right py-3 pr-4">PSNR ↑ (dB)</th>
                  <th className="text-right py-3 pr-4">SSIM ↑</th>
                  <th className="text-right py-3 pr-4">Razão ↑</th>
                  <th className="text-right py-3 pr-4">Redução BW</th>
                  <th className="text-right py-3">Bruto → Latente</th>
                </tr>
              </thead>
              <tbody>
                {okResults.map((row) => (
                  <tr
                    key={`${row.dataset}-${row.model}`}
                    className="border-b border-[#121c2e] hover:bg-[#0a111b] transition"
                  >
                    <td className="py-3 pr-4 font-bold" style={{ color: DATASET_COLORS[row.dataset] }}>
                      {row.dataset.toUpperCase()}
                    </td>
                    <td className="py-3 pr-4 text-slate-300">{row.model}</td>
                    <td className="py-3 pr-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          row.weights_loaded
                            ? "bg-[#073529] text-neon border border-neon"
                            : "bg-[#3b1a1a] text-[#ff9a9a] border border-[#ff7b7b]"
                        }`}
                      >
                        {row.weights_loaded ? "Treinado" : "Demo"}
                      </span>
                    </td>
                    <td className="text-right pr-4 text-slate-200">{r(row.mse_mean, 5)}</td>
                    <td className="text-right pr-4 text-[#ffd166]">{r(row.psnr_mean, 1)} dB</td>
                    <td className="text-right pr-4 text-[#489dff]">{r(row.ssim_mean, 3)}</td>
                    <td className="text-right pr-4 text-neon font-bold">{r(row.compression_ratio_mean, 1)}×</td>
                    <td className="text-right pr-4 text-neon">{row.bandwidth_reduction_pct}%</td>
                    <td className="text-right text-slate-400">
                      {fmtBytes(row.original_bytes)} → {fmtBytes(row.latent_bytes)}{" "}
                      <span className="text-xs text-slate-500">({bits}b)</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Charts ────────────────────────────────────────────────────────── */}
      {okResults.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          {/* Compression ratio bar chart */}
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
              Razão de Compressão por Dataset × Modelo
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={compressionBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis
                    dataKey="name"
                    stroke="#8a9ab4"
                    style={{ fontSize: "10px", fontFamily: "monospace" }}
                    interval={0}
                    tick={{ fill: "#8a9ab4" }}
                  />
                  <YAxis
                    stroke="#8a9ab4"
                    style={{ fontSize: "10px", fontFamily: "monospace" }}
                    label={{ value: "Razão (×)", angle: -90, position: "insideLeft", fill: "#8a9ab4" }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }}
                    formatter={(v) => [`${v}×`, "Compressão"]}
                  />
                  <Bar dataKey="ratio" radius={[4, 4, 0, 0]}>
                    {compressionBarData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* PSNR + SSIM quality bar chart */}
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
              Qualidade Semântica (PSNR + SSIM×100)
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qualityBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis
                    dataKey="name"
                    stroke="#8a9ab4"
                    style={{ fontSize: "10px", fontFamily: "monospace" }}
                  />
                  <YAxis stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontFamily: "monospace", fontSize: "11px" }} />
                  <Bar dataKey="PSNR" name="PSNR (dB)" fill="#ffd166" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="SSIM" name="SSIM×100" fill="#489dff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Radar chart: VAE vs AE on first dataset */}
          {vaeResult && aeResult && (
            <div className="rounded-xl border border-line bg-panel p-6 md:col-span-2">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-1">
                VAE vs AE — Análise Multidimensional ({vaeResult.dataset.toUpperCase()})
              </h3>
              <p className="text-xs text-slate-400 font-mono mb-4">
                Cada eixo representa uma métrica normalizada. VAE tende a superar AE em robustez semântica.
              </p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#223046" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: "#8a9ab4", fontSize: 11, fontFamily: "monospace" }} />
                    <PolarRadiusAxis stroke="#223046" tick={{ fill: "#8a9ab4", fontSize: 9 }} />
                    <Radar name="VAE" dataKey="VAE" stroke="#00f6a2" fill="#00f6a2" fillOpacity={0.3} />
                    <Radar name="AE"  dataKey="AE"  stroke="#ff7b7b" fill="#ff7b7b" fillOpacity={0.3} />
                    <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "12px" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Scalability Table ─────────────────────────────────────────────── */}
      {okResults.length > 0 && <ScalabilityTable results={okResults} />}

      {/* ── Methodology note ──────────────────────────────────────────────── */}
      {okResults.length > 0 && (
        <div className="rounded-xl border border-[#1a2537] bg-[#050c16] p-5 font-mono">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
            Nota Metodológica
          </h3>
          <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
            <li>
              Cada combinação (dataset × modelo) avalia <strong>{numSamples} imagens</strong> do conjunto de teste com seed fixo 42 para reprodutibilidade.
            </li>
            <li>
              Quantização: <strong>{bits} bits</strong> (uniform min-max). Transmissão real: latente (int{bits}) + 4 bytes (escala float32).
            </li>
            <li>
              SSIM implementado por janela Gaussiana 11×11 (σ=1.5), conforme Wang et al. (2004). Faixa: [-1, 1].
            </li>
            <li>
              Instâncias marcadas como <span className="text-[#ff9a9a]">Demo</span> não carregaram pesos treinados —
              execute <code className="text-neon">python -m app.train_local</code> para pesos reais.
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
