export async function registerResultsRoutes(app, mlServiceUrl) {
  app.get("/api/results/latest", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/results/latest`);
    const payload = await response.text();

    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(payload);
  });

  app.get("/api/results/experiments", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/results/experiments`);
    const payload = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(payload);
  });

  app.get("/api/results/experiments/:experimentId", async (request, reply) => {
    const { experimentId } = request.params;
    const response = await fetch(`${mlServiceUrl}/results/experiments/${encodeURIComponent(experimentId)}`);
    const payload = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(payload);
  });

  app.get("/api/results/artifact/:experimentId/*", async (request, reply) => {
    const { experimentId } = request.params;
    const wildcardPath = request.params["*"];
    const response = await fetch(
      `${mlServiceUrl}/results/artifact/${encodeURIComponent(experimentId)}/${wildcardPath}`
    );

    if (!response.body) {
      const text = await response.text();
      reply.code(response.status).send(text);
      return;
    }

    reply.raw.writeHead(response.status, {
      "Content-Type": response.headers.get("content-type") || "application/octet-stream",
      "Cache-Control": "no-cache",
    });

    const reader = response.body.getReader();
    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        reply.raw.write(Buffer.from(value));
      }
      reply.raw.end();
    };

    pump().catch(() => reply.raw.end());
    return reply;
  });
}
