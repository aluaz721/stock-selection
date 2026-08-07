import type { BenchmarkSummary } from "@/lib/types";

export function MetricCard({ label, data, highlight }: { label: string; data: BenchmarkSummary; highlight?: boolean }) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        highlight ? "border-foreground/20 bg-foreground/[0.03]" : "border-foreground/10"
      }`}
    >
      <div className="text-sm font-medium text-foreground/60">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">
        {(data.annualized_return * 100).toFixed(1)}%{" "}
        <span className="text-base font-normal text-foreground/50">/ yr</span>
      </div>
      <div className="mt-1 text-sm text-foreground/60 tabular-nums">Sharpe {data.sharpe_ratio.toFixed(2)}</div>
      <div className="text-sm text-foreground/50 tabular-nums">Max drawdown {(data.max_drawdown * 100).toFixed(1)}%</div>
    </div>
  );
}
