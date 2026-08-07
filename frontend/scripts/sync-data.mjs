// Copies the pipeline's prediction output (../data/predictions/*.json,
// regenerated daily by .github/workflows/daily-selection.yml at the repo
// root) into public/data/ so the Next.js app has a stable local path to
// read at build time, regardless of where in the repo tree the actual
// pipeline output lives. Runs automatically via predev/prebuild.

import { existsSync, mkdirSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SOURCE_DIR = join(__dirname, "..", "..", "data", "predictions");
const DEST_DIR = join(__dirname, "..", "public", "data");
const FILES = ["latest.json", "history.json"];

mkdirSync(DEST_DIR, { recursive: true });

for (const file of FILES) {
  const src = join(SOURCE_DIR, file);
  if (!existsSync(src)) {
    console.error(`sync-data: missing ${src} -- run \`python -m app.reporting.build_frontend_data\` first`);
    process.exit(1);
  }
  copyFileSync(src, join(DEST_DIR, file));
}

console.log(`sync-data: copied ${FILES.join(", ")} from ${SOURCE_DIR} to ${DEST_DIR}`);
