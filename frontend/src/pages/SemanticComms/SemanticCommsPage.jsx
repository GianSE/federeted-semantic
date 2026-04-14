import { useState } from "react";

export default function SemanticCommsPage() {
  const [dataset, setDataset] = useState("fashion");
  const [modelType, setModelType] = useState("cnn_vae");
  const [loadingProcess, setLoadingProcess] = useState(false);
  const [processResult, setProcessResult] = useState(null);

  const [maskType, setMaskType] = useState("Metade Inferior");
  const [maskRatio, setMaskRatio] = useState(0.5);
  const [loadingComplete, setLoadingComplete] = useState(false);
  const [completeResult, setCompleteResult] = useState(null);

  async function handleProcess() {
    setLoadingProcess(true);
    try {
      const response = await fetch("/api/semantic/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, model_type: modelType }),
      });
      const data = await response.json();
      setProcessResult(data);
    } catch (err) {
      console.error(err);
    }
    setLoadingProcess(false);
  }

  async function handleComplete() {
    setLoadingComplete(true);
    try {
      const response = await fetch("/api/semantic/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, model_type: modelType, mask_type: maskType, mask_ratio: maskRatio }),
      });
      const data = await response.json();
      setCompleteResult(data);
    } catch (err) {
      console.error(err);
    }
    setLoadingComplete(false);
  }

  function renderImage(tensorData, label) {
    if (!tensorData) return null;
    let isRGB = false;
    let height = tensorData.length;
    let width = tensorData[0]?.length || 28;
    
    // Suporte ao CIFAR-10 RGB (Tensor retorna [3, H, W])
    if (tensorData.length === 3 && Array.isArray(tensorData[0][0])) {
       isRGB = true;
       height = tensorData[0].length;
       width = tensorData[0][0].length;
    }

    const pixels = [];
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
         if (isRGB) {
             const r = Math.floor(Math.max(0, Math.min(1, tensorData[0][i][j])) * 255);
             const g = Math.floor(Math.max(0, Math.min(1, tensorData[1][i][j])) * 255);
             const b = Math.floor(Math.max(0, Math.min(1, tensorData[2][i][j])) * 255);
             pixels.push(<div key={`${i}-${j}`} style={{ backgroundColor: `rgb(${r},${g},${b})`, width: 4, height: 4 }} />);
         } else {
             const val = Math.floor(Math.max(0, Math.min(1, tensorData[i][j])) * 255);
             pixels.push(<div key={`${i}-${j}`} style={{ backgroundColor: `rgb(${val},${val},${val})`, width: 4, height: 4 }} />);
         }
      }
    }

    return (
      <div className="flex flex-col items-center">
        <div 
          className="grid gap-[1px] bg-slate-800 p-[1px] rounded"
          style={{ gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))` }}
        >
          {pixels}
        </div>
        {label !== undefined && <span className="mt-2 text-xs text-slate-400 font-mono">Classe: {label}</span>}
      </div>
    );
  }

  return (
    <div className="grid gap-6">
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-2">Comunicação Semântica (Compressão IA)</h2>
        <p className="text-sm text-slate-400 mb-6">Testbed de conversão de imagens brutas em vetores semânticos quantizados para transmissão ultraleve.</p>
        
        <div className="flex gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 max-w-xs">
            <label className="block mb-2">Dataset Base</label>
            <select className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2" value={dataset} onChange={e => setDataset(e.target.value)}>
              <option value="fashion">Fashion-MNIST</option>
              <option value="mnist">MNIST Clássico</option>
              <option value="cifar10">CIFAR-10 Colorido (32x32)</option>
            </select>
          </div>
          <div className="flex-1 max-w-xs">
            <label className="block mb-2">IA Encoder</label>
            <select className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2" value={modelType} onChange={e => setModelType(e.target.value)}>
              <option value="cnn_vae">GenAI Variational AE (VAE)</option>
              <option value="cnn_ae">Autoencoder Clássico (AE)</option>
            </select>
          </div>
        </div>

        <button 
          onClick={handleProcess}
          disabled={loadingProcess}
          className="rounded-md bg-[#073529] border border-neon text-neon px-4 py-2 font-mono text-sm hover:bg-[#0b2a22] transition"
        >
          {loadingProcess ? "Processando..." : "Gerar Vetor Latente & Reconstruir"}
        </button>

        {processResult?.status === "ok" && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="text-center bg-[#0a111b] p-4 rounded border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Imagem Bruta Original</h3>
              {renderImage(processResult.original, processResult.label)}
              <p className="mt-4 text-xs text-slate-500 font-mono">Payload: 3136 Bytes (Float32)</p>
            </div>
            
            <div className="text-center font-mono text-sm rounded bg-[#0b2a22] p-4 border border-neon">
              <p className="text-neon mb-2">📡 Transmissão Simulada</p>
              <p className="text-slate-300">Latência Evitada</p>
              <div className="my-4 text-3xl font-bold text-white">
                 {(3136 / processResult.latent_size_int8).toFixed(1)}x
              </div>
              <p className="text-xs text-slate-400">Menos dados enviados</p>
              <div className="mt-4 pt-4 border-t border-[#073529] text-left grid gap-1">
                <p><span className="text-slate-500">Int8 Payload:</span> <span className="text-green-400">{processResult.latent_size_int8} Bytes</span></p>
                <p><span className="text-slate-500">Float Payload:</span> {processResult.latent_size_float} Bytes</p>
              </div>
            </div>

            <div className="text-center bg-[#0a111b] p-4 rounded border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Reconstrução da IA</h3>
              {renderImage(processResult.reconstructed)}
              <div className="mt-4 pt-2 text-xs font-mono grid gap-1 text-slate-400">
                <p>PSNR: {(processResult.psnr).toFixed(1)} dB</p>
                <p>MSE: {(processResult.mse).toFixed(4)}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-2">Completação de Falhas de Canal (Masking)</h2>
        <p className="text-sm text-slate-400 mb-6">Envia partes corrompidas de uma foto na rede e avalia a completação feita pela IA hospedeira.</p>

        <div className="flex gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 max-w-xs">
            <label className="block mb-2">Tipo de Corrupção (Drop)</label>
            <select className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2" value={maskType} onChange={e => setMaskType(e.target.value)}>
              <option value="Metade Inferior">Cortar Metade Inferior</option>
              <option value="Metade Direita">Cortar Metade Direita</option>
              <option value="Pixels Aleatórios">Ruído de Pixel Aleatório</option>
            </select>
          </div>
          <div className="flex-1 max-w-xs">
            <label className="block mb-2">Severidade ({Math.round(maskRatio*100)}% descartado)</label>
            <input type="range" min="0.1" max="0.9" step="0.1" value={maskRatio} onChange={e => setMaskRatio(Number(e.target.value))} className="w-full mt-2" />
          </div>
        </div>

        <button 
          onClick={handleComplete}
          disabled={loadingComplete}
          className="rounded-md bg-[#3b1a1a] border border-[#ff7b7b] text-[#ff9a9a] px-4 py-2 font-mono text-sm hover:bg-[#2c1313] transition"
        >
          {loadingComplete ? "Simulando Queda..." : "Transmitir Imagem Quebrada"}
        </button>

        {completeResult?.status === "ok" && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="text-center bg-[#0a111b] p-4 rounded border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Mídia Original Completa</h3>
              {renderImage(completeResult.original)}
            </div>
            
            <div className="text-center bg-[#1a0f0f] border border-[#522929] p-4 rounded">
              <h3 className="text-sm font-mono text-[#ff9a9a] mb-4">Captado no Destino</h3>
              {renderImage(completeResult.masked)}
            </div>

            <div className="text-center bg-[#0a111b] p-4 rounded border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Restaurado pela GenAI</h3>
              {renderImage(completeResult.completed)}
              <div className="mt-4 pt-2 text-xs font-mono grid gap-1 text-slate-400">
                <p>PSNR Original vs Restore: {(completeResult.psnr).toFixed(1)} dB</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
