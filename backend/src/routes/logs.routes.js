export async function registerLogRoutes(app, mlServiceUrl) {
  app.get("/api/logs/stream", async (request, reply) => {
    const query = request.query || {};
    const target = typeof query.target === "string" ? query.target : "server";
    const upstream = await fetch(`${mlServiceUrl}/training/logs/stream?target=${encodeURIComponent(target)}`);

    reply.raw.writeHead(upstream.status, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    if (!upstream.body) {
      reply.raw.end();
      return reply;
    }

    const reader = upstream.body.getReader();

    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        reply.raw.write(Buffer.from(value));
      }
      reply.raw.end();
    };

    pump().catch(() => reply.raw.end());
    return reply;
  });
}
