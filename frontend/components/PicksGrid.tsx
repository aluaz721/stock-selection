import { BUCKET_ORDER, formatBucketName } from "@/lib/data";
import type { LatestData } from "@/lib/types";

export function PicksGrid({ buckets }: { buckets: LatestData["buckets"] }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {BUCKET_ORDER.map((bucket) => (
        <div key={bucket} className="rounded-lg border border-foreground/10 p-4">
          <div className="text-sm font-medium">{formatBucketName(bucket)}</div>
          <ul className="mt-2 space-y-1">
            {(buckets[bucket] ?? []).map((pick) => (
              <li key={pick.tic} className="flex items-center justify-between text-sm tabular-nums">
                <span className="font-mono">{pick.tic}</span>
                <span className={pick.predicted_return >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
                  {pick.predicted_return >= 0 ? "+" : ""}
                  {(pick.predicted_return * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
