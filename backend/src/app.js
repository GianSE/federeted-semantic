import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import { registerTrainingRoutes } from "./routes/training.routes.js";
import { registerResultsRoutes } from "./routes/results.routes.js";
import { registerLogRoutes } from "./routes/logs.routes.js";
import { registerSemanticRoutes } from "./routes/semantic.routes.js";
import { registerBenchmarkRoutes } from "./routes/benchmark.routes.js";

const app = Fastify({ logger: true });
const port = Number(process.env.PORT || 3000);
const mlServiceUrl = process.env.ML_SERVICE_URL || "http://ml-service:8000";

await app.register(cors, { origin: true });
await app.register(multipart, {
  limits: {
    fileSize: 100 * 1024 * 1024,
    files: 20,
  },
});

app.get("/health", async () => ({ status: "ok", service: "backend" }));

await registerTrainingRoutes(app, mlServiceUrl);
await registerResultsRoutes(app, mlServiceUrl);
await registerLogRoutes(app, mlServiceUrl);
await registerSemanticRoutes(app, mlServiceUrl);
await registerBenchmarkRoutes(app, mlServiceUrl);

app.listen({ port, host: "0.0.0.0" }).catch((err) => {
  app.log.error(err);
  process.exit(1);
});
