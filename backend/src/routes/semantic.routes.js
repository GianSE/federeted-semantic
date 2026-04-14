export async function registerSemanticRoutes(app, mlServiceUrl) {
  app.post("/api/semantic/process", async (request, reply) => {
    try {
      const response = await fetch(`${mlServiceUrl}/semantic/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body || {}),
      });
      if (!response.ok) {
        throw new Error(`ML Service falhou com status ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      app.log.error(error);
      return reply.code(500).send({ error: error.message });
    }
  });

  app.post("/api/semantic/complete", async (request, reply) => {
    try {
      const response = await fetch(`${mlServiceUrl}/semantic/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body || {}),
      });
      if (!response.ok) {
        throw new Error(`ML Service falhou com status ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      app.log.error(error);
      return reply.code(500).send({ error: error.message });
    }
  });

  app.post("/api/semantic/tradeoff", async (request, reply) => {
    try {
      const response = await fetch(`${mlServiceUrl}/semantic/tradeoff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body || {}),
      });
      if (!response.ok) {
        throw new Error(`ML Service falhou com status ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      app.log.error(error);
      return reply.code(500).send({ error: error.message });
    }
  });
}
