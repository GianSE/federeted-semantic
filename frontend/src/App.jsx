import { NavLink, Route, Routes } from "react-router-dom";
import TrainingDashboardPage from "./pages/TrainingDashboard/TrainingDashboardPage";
import ResultsPage from "./pages/Results/ResultsPage";
import SemanticCommsPage from "./pages/SemanticComms/SemanticCommsPage";
import TradeoffPage from "./pages/Tradeoff/TradeoffPage";

const links = [
  { to: "/", label: "Treinamento Federado" },
  { to: "/results", label: "Relatórios de Resultados" },
  { to: "/semantic", label: "Teste Semântico (GenAI)" },
  { to: "/tradeoff", label: "Análise Trade-off (SNR/Bits)" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-text lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-r border-line bg-[#070d14] p-4 lg:min-h-screen">
        <h1 className="font-mono text-xl font-semibold tracking-wide text-neon">Lab Federado</h1>
        <p className="mt-1 font-mono text-xs text-slate-400">MNIST/CIFAR - Comunicação Semântica</p>

        <nav className="mt-6 grid gap-2 font-mono text-sm">
          {links.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md border px-3 py-2 transition ${
                  isActive ? "border-neon bg-[#0b2a22] text-neon" : "border-line bg-[#0d1420] text-slate-300"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="p-4 lg:p-6">
        <Routes>
          <Route path="/" element={<TrainingDashboardPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/semantic" element={<SemanticCommsPage />} />
          <Route path="/tradeoff" element={<TradeoffPage />} />
        </Routes>
      </main>
    </div>
  );
}
