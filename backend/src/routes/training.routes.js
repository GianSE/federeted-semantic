export async function registerTrainingRoutes(app, mlServiceUrl) {
  app.post("/api/training/start", async (request, reply) => {
    const payload = request.body || {};
    const response = await fetch(`${mlServiceUrl}/training/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/training/status", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/training/status`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/training/pause", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/training/pause`, { method: "POST" });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/training/resume", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/training/resume`, { method: "POST" });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/training/stop", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/training/stop`, { method: "POST" });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/training/logs/clear", async (request, reply) => {
    const payload = request.body || {};
    const response = await fetch(`${mlServiceUrl}/training/logs/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });
}
