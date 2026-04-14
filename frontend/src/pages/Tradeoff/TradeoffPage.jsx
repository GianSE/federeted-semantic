import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";

export default function TradeoffPage() {
  const [dataset, setDataset] = useState("fashion");
  const [modelType, setModelType] = useState("cnn_vae");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  async function handleSimulate() {
    setLoading(true);
    try {
      const response = await fetch("/api/semantic/tradeoff", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          model_type: modelType,
          num_samples: 10, // Aumentando a média
          snr_levels: [0.0, 5.0, 10.0, 15.0, 20.0],
          bits_levels: [4, 8, 16, 32]
        })
      });
      const data = await response.json();
      if (data.status === "ok") {
        setResults(data.items);
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  }

  // Grupar dados para exibição:
  // Queremos um gráfico X: SNR (dB), Y: PSNR. Linhas representam Bits.
  const chartData = [];
  if (results.length > 0) {
    const snrs = [...new Set(results.map(r => r.snr_db))].sort((a,b)=>a-b);
    for (const snr of snrs) {
      const point = { snr_db: snr };
      const subset = results.filter(r => r.snr_db === snr);
      for (const bits of [4, 8, 16, 32]) {
        const matches = subset.filter(r => r.bits === bits);
        const avg_psnr = matches.reduce((acc, curr) => acc + curr.psnr, 0) / matches.length;
        point[`bits_${bits}`] = avg_psnr;
      }
      chartData.push(point);
    }
  }

  // Gráfico de Payload: Barras horizontais comparativas de bytes
  let payload4 = 0, payload8 = 0, payload32 = 0;
  if (results.length > 0) {
    payload4 = results.find(r => r.bits === 4)?.payload_bytes || 0;
    payload8 = results.find(r => r.bits === 8)?.payload_bytes || 0;
    payload32 = results.find(r => r.bits === 32)?.payload_bytes || 0;
  }
  const barData = [
    { name: "Flutuante (32b)", bytes: payload32, fill: "#ff7b7b" },
    { name: "Quant. Sutil (8b)", bytes: payload8, fill: "#ffd166" },
    { name: "Quant. Extrema (4b)", bytes: payload4, fill: "#00f6a2" },
  ];

  return (
    <div className="grid gap-6">
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-2">Análise: Taxa-Distorção (Trade-off GenAI)</h2>
        <p className="text-sm text-slate-400 mb-6">Analise como a IA Generativa (Layer Semântica) comporta-se variando os Níveis de Ruído na Rede (SNR) e a Agressividade da Quantização nos Gêmeos Digitais.</p>
        
        <div className="flex gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 max-w-xs">
            <label className="block mb-2">Dataset Base</label>
            <select className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2" value={dataset} onChange={e => setDataset(e.target.value)}>
              <option value="fashion">Fashion-MNIST</option>
              <option value="mnist">MNIST</option>
              <option value="cifar10">CIFAR-10 (Colorido)</option>
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
          onClick={handleSimulate}
          disabled={loading}
          className="rounded-md bg-[#073529] border border-neon text-neon px-4 py-2 font-mono text-sm hover:bg-[#0b2a22] transition"
        >
          {loading ? "Rodando Monte Carlo..." : "Simular Extresso de Canal & Quantização"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          {/* Gráfico Linear PSNR vs SNR */}
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold text-slate-300 mb-4 font-mono uppercase tracking-wide">Qualidade (PSNR) vs Canal (SNR)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
               <LineChart data={chartData}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                 <XAxis dataKey="snr_db" stroke="#9aa7bd" label={{ value: 'Sinal/Ruído (dB)', position: 'insideBottom', offset: -5 }} />
                 <YAxis stroke="#9aa7bd" label={{ value: 'PSNR', angle: -90, position: 'insideLeft' }} />
                 <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d" }} />
                 <Legend verticalAlign="top" height={36}/>
                 <Line type="monotone" name="Float32" dataKey="bits_32" stroke="#ff7b7b" strokeWidth={2} dot={true} />
                 <Line type="monotone" name="Int16" dataKey="bits_16" stroke="#489dff" strokeWidth={2} dot={true} />
                 <Line type="monotone" name="Int8" dataKey="bits_8" stroke="#ffd166" strokeWidth={3} dot={true} />
                 <Line type="monotone" name="Int4" dataKey="bits_4" stroke="#00f6a2" strokeWidth={2} dot={true} />
               </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-4 text-xs text-slate-400 font-mono">Observe como a quantização Extrema (Int4 e Int8) praticamente empatam com os floats quando o canal sofre muito ruído AWGN (0 a 5 dB). A Rede Neural absorve a matemática sem precisar dos metadados longos.</p>
          </div>

          {/* Gráfico Bar Payload */}
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold text-slate-300 mb-4 font-mono uppercase tracking-wide">Tamanho Físico Tráfego Semântico</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis type="number" stroke="#9aa7bd" label={{ value: 'Bytes no Canal', position: 'insideBottom', offset: -5 }} />
                  <YAxis dataKey="name" type="category" stroke="#9aa7bd" width={110} />
                  <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d" }} />
                  <Bar dataKey="bytes" radius={[0, 4, 4, 0]}>
                    {barData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-4 text-xs text-slate-400 font-mono">
              Comprovado! A quantização de Int4 reduzimos <span className="text-neon font-bold">{(100 - (payload4/payload32)*100).toFixed(1)}%</span> da banda passante sacrificando pouquíssimos dBs na Imagem acima.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
