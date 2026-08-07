"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SeriesPoint } from "@/lib/types";

const SERIES = [
  { key: "strategy", name: "Strategy (top-3/bucket)", color: "#2563eb" },
  { key: "spy", name: "SPY", color: "#9ca3af" },
  { key: "qqq", name: "QQQ", color: "#d1d5db" },
] as const;

// history.json's series is 2306 daily points -- decimate for the chart
// (SVG with 2306 * 3 points renders fine, but a coarser line reads better
// at typical dashboard widths and keeps tooltip hit-testing responsive).
function decimate<T>(rows: T[], targetPoints = 400): T[] {
  const step = Math.max(1, Math.floor(rows.length / targetPoints));
  return rows.filter((_, i) => i % step === 0 || i === rows.length - 1);
}

// One tick per calendar year, at that year's first available date --
// Recharts' auto tick placement on a categorical (string) axis spaces
// ticks by pixel gap, not by parsing dates, so it can otherwise land two
// ticks in the same year and label them identically.
function yearTicks(rows: SeriesPoint[]): string[] {
  const ticks: string[] = [];
  let lastYear = "";
  for (const row of rows) {
    const year = row.date.slice(0, 4);
    if (year !== lastYear) {
      ticks.push(row.date);
      lastYear = year;
    }
  }
  return ticks;
}

export function PerformanceChart({ series }: { series: SeriesPoint[] }) {
  const data = decimate(series);
  const ticks = yearTicks(data);

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
        <XAxis
          dataKey="date"
          ticks={ticks}
          tickFormatter={(d: string) => d.slice(0, 4)}
          tick={{ fontSize: 12, fill: "currentColor", opacity: 0.6 }}
        />
        <YAxis
          tickFormatter={(v: number) => `${v.toFixed(1)}x`}
          tick={{ fontSize: 12, fill: "currentColor", opacity: 0.6 }}
          width={44}
        />
        <Tooltip
          formatter={(value, name) => [`${Number(value).toFixed(3)}x`, name]}
          labelFormatter={(label) => (typeof label === "string" ? label : "")}
          contentStyle={{ fontSize: 13, borderRadius: 8 }}
        />
        <Legend wrapperStyle={{ fontSize: 13 }} />
        {SERIES.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
