export interface BenchmarkSummary {
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

export interface SeriesPoint {
  date: string;
  strategy: number;
  spy: number;
  qqq: number;
}

export interface HistoryData {
  generated_at: string;
  config: {
    top_n_per_bucket: number;
    start: string;
    end: string;
  };
  summary: {
    strategy: BenchmarkSummary;
    spy: BenchmarkSummary;
    qqq: BenchmarkSummary;
  };
  series: SeriesPoint[];
}

export interface Pick {
  tic: string;
  predicted_return: number;
}

export interface LatestData {
  as_of: string;
  vintage_breakdown: Record<string, number>;
  val_cutoff: string;
  generated_at: string;
  buckets: Record<string, Pick[]>;
}
