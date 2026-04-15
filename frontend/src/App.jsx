import { NavLink, Route, Routes } from "react-router-dom";
import TrainingDashboardPage from "./pages/TrainingDashboard/TrainingDashboardPage";
import ResultsPage from "./pages/Results/ResultsPage";
import SemanticCommsPage from "./pages/SemanticComms/SemanticCommsPage";
import TradeoffPage from "./pages/Tradeoff/TradeoffPage";
import BenchmarkPage from "./pages/Benchmark/BenchmarkPage";

/**
 * Navigation links for the main sidebar.
 * Dead / empty routes (KnowledgeBase, PaperCopilot, PaperRAG) have been removed.
 */
const links = [
  { to: "/",         label: "Treinamento Federado",        icon: "⚙" },
  { to: "/results",  label: "Relatórios de Resultados",    icon: "📊" },
  { to: "/semantic", label: "Comunicação Semântica",        icon: "📡" },
  { to: "/tradeoff", label: "Análise Trade-off (SNR/Bits)", icon: "📉" },
  { to: "/benchmark",label: "Benchmark Multi-Dataset",     icon: "🔬" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-text lg:grid lg:grid-cols-[270px_1fr]">
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="border-r border-line bg-[#070d14] p-5 lg:min-h-screen flex flex-col">
        <div>
          <h1 className="font-mono text-xl font-semibold tracking-wide text-neon">
            Lab Semântico
          </h1>
          <p className="mt-1 font-mono text-xs text-slate-400 leading-relaxed">
            MNIST · Fashion · CIFAR-10
            <br />
            Comunicação Semântica + FedAvg
          </p>
        </div>

        <nav className="mt-6 grid gap-1.5 font-mono text-sm flex-1">
          {links.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `rounded-md border px-3 py-2.5 transition flex items-center gap-2 ${
                  isActive
                    ? "border-neon bg-[#0b2a22] text-neon"
                    : "border-line bg-[#0d1420] text-slate-300 hover:text-slate-100 hover:bg-[#111c30]"
                }`
              }
            >
              <span className="text-base leading-none">{item.icon}</span>
              <span className="text-xs leading-snug">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-line text-[10px] font-mono text-slate-500 leading-relaxed">
          <p>IC — Comunicação Semântica</p>
          <p>UTFPR — Cornélio Procópio</p>
          <p className="mt-1 text-[9px] text-slate-600">
            Hipótese: Vetores latentes reduzem banda preservando semântica.
          </p>
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className="p-4 lg:p-6 min-w-0">
        <Routes>
          <Route path="/"          element={<TrainingDashboardPage />} />
          <Route path="/results"   element={<ResultsPage />} />
          <Route path="/semantic"  element={<SemanticCommsPage />} />
          <Route path="/tradeoff"  element={<TradeoffPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
        </Routes>
      </main>
    </div>
  );
}
