import { useEffect, useRef, useState } from "react";

// ─── Dataset metadata (must match DATASET_META in image_utils.py) ────────────

const DATASET_META = {
  fashion: { channels: 1, height: 28, width: 28, rawBytes: 3136,  label: "Fashion-MNIST" },
  mnist:   { channels: 1, height: 28, width: 28, rawBytes: 3136,  label: "MNIST Clássico" },
  cifar10: { channels: 3, height: 32, width: 32, rawBytes: 12288, label: "CIFAR-10 Colorido (32×32)" },
};

// ─── Canvas-based image renderer ─────────────────────────────────────────────

/**
 * Render a tensor returned by the Python backend onto an HTML canvas.
 *
 * The backend returns tensors squeezed of the batch dim:
 *   - Grayscale: [H, W]         (28×28 or 32×32)
 *   - RGB:       [3, H, W]      (CIFAR-10)
 *
 * @param {Array} tensorData  - Nested JavaScript array from JSON response.
 * @param {number} displaySize - Canvas render width/height in px (default 140).
 */
function TensorImage({ tensorData, label, displaySize = 140 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!tensorData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const isRGB = (
      tensorData.length === 3 &&
      Array.isArray(tensorData[0]) &&
      Array.isArray(tensorData[0][0])
    );

    const height = isRGB ? tensorData[0].length : tensorData.length;
    const width  = isRGB ? tensorData[0][0].length : (tensorData[0]?.length || height);

    canvas.width  = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const offset = (y * width + x) * 4;
        let r, g, b;

        if (isRGB) {
          r = Math.round(Math.max(0, Math.min(1, tensorData[0][y][x])) * 255);
          g = Math.round(Math.max(0, Math.min(1, tensorData[1][y][x])) * 255);
          b = Math.round(Math.max(0, Math.min(1, tensorData[2][y][x])) * 255);
        } else {
          const v = Math.round(Math.max(0, Math.min(1, tensorData[y][x])) * 255);
          r = g = b = v;
        }

        imageData.data[offset]     = r;
        imageData.data[offset + 1] = g;
        imageData.data[offset + 2] = b;
        imageData.data[offset + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [tensorData]);

  if (!tensorData) return null;

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        style={{
          width: displaySize,
          height: displaySize,
          imageRendering: "pixelated",
          borderRadius: "4px",
          border: "1px solid #1d2a3d",
        }}
      />
      {label !== undefined && (
        <span className="text-xs text-slate-400 font-mono">Classe: {label}</span>
      )}
    </div>
  );
}

// ─── Metric badge helper ──────────────────────────────────────────────────────

function MetricRow({ label, value, color = "text-slate-200", unit = "" }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-[#121c2e] last:border-0">
      <span className="text-xs text-slate-500 font-mono">{label}</span>
      <span className={`text-xs font-bold font-mono ${color}`}>
        {value}{unit && <span className="text-slate-400 ml-1">{unit}</span>}
      </span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

/**
 * SemanticCommsPage
 *
 * Demonstrates two semantic communication scenarios:
 *   1. Compression pipeline: raw image → latent vector → reconstruction
 *   2. Channel error recovery: masked (corrupted) image → AI-based completion
 *
 * Metrics shown: MSE, PSNR (dB), SSIM, compression ratio, bandwidth reduction.
 */
export default function SemanticCommsPage() {
  const [dataset,    setDataset]    = useState("fashion");
  const [modelType,  setModelType]  = useState("cnn_vae");
  const [bits,       setBits]       = useState(8);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [processErr, setProcessErr] = useState(null);

  const [maskType,     setMaskType]     = useState("Metade Inferior");
  const [maskRatio,    setMaskRatio]    = useState(0.5);
  const [loadingComp,  setLoadingComp]  = useState(false);
  const [compResult,   setCompResult]   = useState(null);
  const [completeErr,  setCompleteErr]  = useState(null);

  const meta = DATASET_META[dataset] ?? DATASET_META.fashion;

  // ── Compression pipeline ──────────────────────────────────────────────────

  async function handleProcess() {
    setLoading(true);
    setProcessErr(null);
    try {
      const res = await fetch("/api/semantic/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, model_type: modelType, bits }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Erro no servidor");
      setResult(data);
    } catch (err) {
      setProcessErr(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Masking / completion ──────────────────────────────────────────────────

  async function handleComplete() {
    setLoadingComp(true);
    setCompleteErr(null);
    try {
      const res = await fetch("/api/semantic/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          model_type: modelType,
          mask_type: maskType,
          mask_ratio: maskRatio,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Erro no servidor");
      setCompResult(data);
    } catch (err) {
      setCompleteErr(err.message);
    } finally {
      setLoadingComp(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="grid gap-6">

      {/* ── Section 1: Semantic Compression Pipeline ─────────────────────── */}
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-1">
          Comunicação Semântica — Pipeline de Compressão
        </h2>
        <p className="text-sm text-slate-400 mb-6 font-mono">
          Converte imagens brutas em vetores latentes quantizados para transmissão
          eficiente — comprova a hipótese central da pesquisa.
        </p>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => { setDataset(e.target.value); setResult(null); }}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              {Object.entries(DATASET_META).map(([key, d]) => (
                <option key={key} value={key}>{d.label}</option>
              ))}
            </select>
          </div>

          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Modelo Encoder</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              <option value="cnn_vae">VAE Convolucional (Recomendado)</option>
              <option value="cnn_ae">Autoencoder Clássico (AE)</option>
            </select>
          </div>

          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Quantização</label>
            <select
              value={bits}
              onChange={(e) => setBits(Number(e.target.value))}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              <option value={4}>Int4 (agressiva)</option>
              <option value={8}>Int8 (equilibrada)</option>
              <option value={16}>Int16 (suave)</option>
              <option value={32}>Float32 (sem quantização)</option>
            </select>
          </div>
        </div>

        <button
          id="semantic-process-btn"
          onClick={handleProcess}
          disabled={loading}
          className="rounded-md bg-[#073529] border border-neon text-neon px-4 py-2 font-mono text-sm hover:bg-[#0b2a22] transition disabled:opacity-50"
        >
          {loading ? "Processando..." : "Gerar Vetor Latente & Reconstruir"}
        </button>

        {processErr && (
          <div className="mt-4 rounded border border-red-800 bg-[#1a0f0f] p-3 text-xs text-red-400 font-mono">
            Erro: {processErr}
          </div>
        )}

        {result?.status === "ok" && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
            {/* Original */}
            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Imagem Bruta Original</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={result.original} label={result.label} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d]">
                <MetricRow label="Tamanho (float32)" value={`${result.original_bytes ?? meta.rawBytes} B`} />
              </div>
            </div>

            {/* Transmission stats */}
            <div className="text-center font-mono text-sm rounded-lg bg-[#0b2a22] p-4 border border-neon">
              <p className="text-neon mb-1 font-bold">📡 Transmissão Semântica</p>
              <p className="text-slate-400 text-xs mb-4">Canal → Receptor</p>

              <div className="text-5xl font-bold text-white my-3">
                {result.compression_ratio != null
                  ? `${result.compression_ratio}×`
                  : `${((result.original_bytes ?? meta.rawBytes) / (result.latent_size_int8 || 1)).toFixed(1)}×`}
              </div>
              <p className="text-xs text-slate-400 mb-4">compressão</p>

              <div className="bg-[#052217] rounded-md p-3 text-left grid gap-1.5">
                <MetricRow
                  label={`Latente Int${bits}`}
                  value={`${result.latent_size_int8} B`}
                  color="text-neon"
                />
                <MetricRow
                  label="Latente Float32"
                  value={`${result.latent_size_float} B`}
                />
                <MetricRow
                  label="Redução de Banda"
                  value={`${result.bandwidth_reduction_pct ?? "—"}%`}
                  color="text-neon"
                />
              </div>
            </div>

            {/* Reconstructed */}
            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Reconstrução da IA</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={result.reconstructed} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d] grid gap-1">
                <MetricRow label="MSE  ↓" value={result.mse?.toFixed(5)} />
                <MetricRow label="PSNR ↑" value={`${result.psnr?.toFixed(1)} dB`} color="text-[#ffd166]" />
                <MetricRow label="SSIM ↑" value={result.ssim?.toFixed(3)} color="text-[#489dff]" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Section 2: Channel Error / Masking ───────────────────────────── */}
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-1">
          Recuperação de Erros de Canal (Masking)
        </h2>
        <p className="text-sm text-slate-400 mb-6 font-mono">
          Simula perda de pacotes enviando apenas parte da imagem, e avalia a
          reconstrução pelo modelo receptor.
        </p>

        <div className="flex flex-wrap gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 min-w-[180px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">
              Tipo de Corrupção
            </label>
            <select
              value={maskType}
              onChange={(e) => setMaskType(e.target.value)}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              <option value="Metade Inferior">Cortar Metade Inferior</option>
              <option value="Metade Direita">Cortar Metade Direita</option>
              <option value="Pixels Aleatórios">Ruído de Pixel Aleatório</option>
            </select>
          </div>

          <div className="flex-1 min-w-[180px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">
              Severidade ({Math.round(maskRatio * 100)}% descartado)
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={maskRatio}
              onChange={(e) => setMaskRatio(Number(e.target.value))}
              className="w-full mt-2 accent-neon"
            />
          </div>
        </div>

        <button
          id="semantic-complete-btn"
          onClick={handleComplete}
          disabled={loadingComp}
          className="rounded-md bg-[#3b1a1a] border border-[#ff7b7b] text-[#ff9a9a] px-4 py-2 font-mono text-sm hover:bg-[#2c1313] transition disabled:opacity-50"
        >
          {loadingComp ? "Simulando Queda de Canal..." : "Transmitir Imagem Quebrada"}
        </button>

        {completeErr && (
          <div className="mt-4 rounded border border-red-800 bg-[#1a0f0f] p-3 text-xs text-red-400 font-mono">
            Erro: {completeErr}
          </div>
        )}

        {compResult?.status === "ok" && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Original Completo</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={compResult.original} label={compResult.label} />
              </div>
            </div>

            <div className="text-center bg-[#1a0f0f] p-4 rounded-lg border border-[#522929]">
              <h3 className="text-sm font-mono text-[#ff9a9a] mb-4">
                Recebido (com queda {Math.round(maskRatio * 100)}%)
              </h3>
              <div className="flex justify-center">
                <TensorImage tensorData={compResult.masked} />
              </div>
            </div>

            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Restaurado pela IA</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={compResult.completed} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d] grid gap-1">
                <MetricRow label="MSE  ↓" value={compResult.mse?.toFixed(5)} />
                <MetricRow label="PSNR ↑" value={`${compResult.psnr?.toFixed(1)} dB`} color="text-[#ffd166]" />
                <MetricRow label="SSIM ↑" value={compResult.ssim?.toFixed(3)} color="text-[#489dff]" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
