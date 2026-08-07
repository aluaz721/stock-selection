import { getHistory, getLatest } from "@/lib/data";
import { MetricCard } from "@/components/MetricCard";
import { PerformanceChart } from "@/components/PerformanceChart";
import { PicksGrid } from "@/components/PicksGrid";

export default function Home() {
  const history = getHistory();
  const latest = getLatest();

  const vintageNote = Object.entries(latest.vintage_breakdown)
    .map(([quarter, count]) => `${count} tickers on ${quarter}`)
    .join(", ");

  return (
    <main className="mx-auto max-w-4xl px-6 py-10 sm:py-14">
      <h1 className="text-2xl font-semibold tracking-tight">ML Stock Selection</h1>

      <section className="mt-8">
        <h2 className="text-sm font-medium text-foreground/60">Backtested performance</h2>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard label={`Strategy (top-${history.config.top_n_per_bucket}/bucket)`} data={history.summary.strategy} highlight />
          <MetricCard label="SPY" data={history.summary.spy} />
          <MetricCard label="QQQ" data={history.summary.qqq} />
        </div>
        <p className="mt-3 text-xs text-foreground/50">
          Backtest window: {history.config.start} to {history.config.end} (2 real market crashes included, not just a bull run).
        </p>

        <div className="mt-4">
          <PerformanceChart series={history.series} />
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-foreground/60">Latest picks</h2>
        <p className="mt-1 text-xs text-foreground/50">
          As of {latest.as_of} (mixed vintage: {vintageNote} — see ML_STOCK_SELECTION.md&apos;s mixed-vintage mode). Model trained
          through {latest.val_cutoff}.
        </p>
        <div className="mt-3">
          <PicksGrid buckets={latest.buckets} />
        </div>
      </section>

      <p className="mt-10 text-xs text-foreground/50">
        Fundamentals are quarterly, filed on scattered dates during earnings season.
      </p>
    </main>
  );
}
