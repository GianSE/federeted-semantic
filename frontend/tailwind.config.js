export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#080b0f",
        panel: "#0f1722",
        line: "#1d2a3d",
        neon: "#00f6a2",
        warn: "#ffd166",
        text: "#d2e1ff",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },
      boxShadow: {
        neon: "0 0 0 1px rgba(0,246,162,0.35), 0 0 18px rgba(0,246,162,0.18)",
      },
    },
  },
  plugins: [],
};
