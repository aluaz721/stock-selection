import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { HistoryData, LatestData } from "./types";

const DATA_DIR = join(process.cwd(), "public", "data");

export function getHistory(): HistoryData {
  return JSON.parse(readFileSync(join(DATA_DIR, "history.json"), "utf-8"));
}

export function getLatest(): LatestData {
  return JSON.parse(readFileSync(join(DATA_DIR, "latest.json"), "utf-8"));
}

export const BUCKET_ORDER = ["growth_tech", "cyclical", "real_assets", "defensive"] as const;

export function formatBucketName(bucket: string): string {
  return bucket
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}
