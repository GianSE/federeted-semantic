/**
 * benchmark.routes.js
 * -------------------
 * Proxy route that forwards cross-dataset benchmark requests to the
 * Python ML service and streams the response back to the browser.
 *
 * Endpoint:
 *   POST /api/experiment/benchmark
 *     Body: { datasets, models, bits, num_samples, seed }
 *     Response: { status, seed, bits, timestamp, results[] }
 *
 *   GET /api/info/architecture
 *     Response: System architecture description JSON
 */

/**
 * Register benchmark-related routes on the Fastify instance.
 *
 * @param {import('fastify').FastifyInstance} app
 * @param {string} mlServiceUrl — Base URL of the Python ML service
 */
export async function registerBenchmarkRoutes(app, mlServiceUrl) {
  /**
   * POST /api/experiment/benchmark
   *
   * Runs a structured cross-dataset quality benchmark and returns aggregate
   * statistics (MSE, PSNR, SSIM, compression ratio) per dataset × model pair.
   */
  app.post("/api/experiment/benchmark", async (request, reply) => {
    try {
      const response = await fetch(`${mlServiceUrl}/experiment/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.body || {}),
      });

      if (!response.ok) {
        const text = await response.text();
        return reply
          .code(response.status)
          .send({ error: `ML Service error: ${text}` });
      }

      const data = await response.json();
      return data;
    } catch (error) {
      app.log.error({ err: error }, "benchmark request failed");
      return reply.code(500).send({ error: error.message });
    }
  });

  /**
   * GET /api/info/architecture
   *
   * Returns a structured description of the system architecture, models,
   * datasets, and metrics for display in the frontend presentation layer.
   */
  app.get("/api/info/architecture", async (request, reply) => {
    try {
      const response = await fetch(`${mlServiceUrl}/info/architecture`);
      if (!response.ok) {
        const text = await response.text();
        return reply.code(response.status).send({ error: text });
      }
      const data = await response.json();
      return data;
    } catch (error) {
      app.log.error({ err: error }, "architecture info request failed");
      return reply.code(500).send({ error: error.message });
    }
  });
}
