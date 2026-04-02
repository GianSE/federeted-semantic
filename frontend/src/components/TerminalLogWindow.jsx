export default function TerminalLogWindow({ logs, title, streamStarted = false }) {
  const content = logs.length
    ? logs.join("\n")
    : streamStarted
      ? "[stream] aguardando novas mensagens..."
      : "[idle] clique em iniciar treinamento para começar a receber logs...";

  return (
    <div className="rounded-xl border border-line bg-black/80 p-3 shadow-neon">
      <div className="mb-2 flex items-center gap-2 text-xs text-slate-400">
        <span className="h-2 w-2 rounded-full bg-red-500" />
        <span className="h-2 w-2 rounded-full bg-yellow-500" />
        <span className="h-2 w-2 rounded-full bg-green-500" />
        <span className="ml-2 font-mono">{title || "training.log"}</span>
      </div>
      <pre className="h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-neon">
        {content}
      </pre>
    </div>
  );
}
