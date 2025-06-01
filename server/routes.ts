import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { Router, Request, Response } from "express";

/**
 * Attaches all REST + websocket routes and returns the underlying Node server
 * so Socket.IO or raw WS can reuse the same port.
 */
export async function registerRoutes(app: import("express").Express): Promise<Server> {
  // ─── Example JSON endpoint ──────────────────────────────────────────────
  const api = Router();
  api.get("/api/health", (_req: Request, res: Response) =>
    res.json({ ok: true, env: process.env.NODE_ENV || "development" })
  );
  app.use(api);

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  // ─── Return raw HTTP server for Vite HMR & WebSockets ───────────────────
  const httpServer = createServer(app);
  return httpServer;
}
