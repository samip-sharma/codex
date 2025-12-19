#!/usr/bin/env python3
"""
Verify George Bayer-style price/time hypotheses using Yahoo Finance data.

Method overview (pragmatic, testable):
- Identify "major bottoms" as local minima within a rolling window.
- Measure whether price targets (+8.75, +17.5, +35, +122.5) are hit
  within a fixed future window from each bottom.
- Check time-based dates (29, 35, 70 calendar days) for a potential
  trend change via sign flip in 5-day returns around the target date.
- Report aggregate hit rates and "price-time square" occurrences
  (price +35 reached within 35 calendar days).

This is a simplified verification; it does not claim causal validity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

PRICE_TARGETS = [8.75, 17.5, 35.0, 122.5]
TIME_TARGETS = [29, 35, 70]


@dataclass
class BottomResult:
    date: pd.Timestamp
    price: float
    price_hits: Dict[float, Optional[int]]
    time_reversals: Dict[int, Optional[bool]]
    square_price_time: bool


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = data[["Adj Close"]].rename(columns={"Adj Close": "close"})
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def find_local_minima(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling_min = df["close"].rolling(window=window * 2 + 1, center=True).min()
    minima = df[df["close"] == rolling_min].copy()
    minima = minima[~minima.index.duplicated(keep="first")]
    return minima


def nearest_trading_day(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    candidates = df.index[df.index >= target_date]
    if candidates.empty:
        return None
    return candidates[0]


def five_day_return(df: pd.DataFrame, end_date: pd.Timestamp) -> Optional[float]:
    window = df.loc[:end_date].tail(6)
    if len(window) < 6:
        return None
    start_price = window["close"].iloc[0]
    end_price = window["close"].iloc[-1]
    return (end_price - start_price) / start_price


def evaluate_bottoms(df: pd.DataFrame, bottoms: pd.DataFrame, forward_days: int) -> List[BottomResult]:
    results: List[BottomResult] = []
    for date, row in bottoms.iterrows():
        bottom_price = float(row["close"])
        price_hits: Dict[float, Optional[int]] = {}
        time_reversals: Dict[int, Optional[bool]] = {}

        forward_window = df.loc[date : date + timedelta(days=forward_days)]
        for target in PRICE_TARGETS:
            target_price = bottom_price + target
            hit = forward_window[forward_window["close"] >= target_price]
            if hit.empty:
                price_hits[target] = None
            else:
                hit_date = hit.index[0]
                price_hits[target] = (hit_date - date).days

        for days in TIME_TARGETS:
            target_date = date + timedelta(days=days)
            trade_date = nearest_trading_day(df, target_date)
            if trade_date is None:
                time_reversals[days] = None
                continue
            prev_return = five_day_return(df, trade_date)
            next_window = df.loc[trade_date:]
            if len(next_window) < 6:
                time_reversals[days] = None
                continue
            next_return = (next_window["close"].iloc[5] - next_window["close"].iloc[0]) / next_window["close"].iloc[0]
            if prev_return is None:
                time_reversals[days] = None
            else:
                time_reversals[days] = np.sign(prev_return) != np.sign(next_return)

        square_price_time = price_hits.get(35.0) is not None and price_hits[35.0] <= 35
        results.append(
            BottomResult(
                date=pd.Timestamp(date),
                price=bottom_price,
                price_hits=price_hits,
                time_reversals=time_reversals,
                square_price_time=square_price_time,
            )
        )
    return results


def summarize(results: List[BottomResult]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    total = len(results)
    summary["bottoms"] = total
    if total == 0:
        return summary

    for target in PRICE_TARGETS:
        hits = [r for r in results if r.price_hits.get(target) is not None]
        summary[f"price_hit_{target}"] = len(hits) / total

    for days in TIME_TARGETS:
        reversals = [r for r in results if r.time_reversals.get(days) is True]
        valid = [r for r in results if r.time_reversals.get(days) is not None]
        summary[f"time_reversal_{days}"] = len(reversals) / len(valid) if valid else 0.0

    squares = [r for r in results if r.square_price_time]
    summary["square_price_time"] = len(squares) / total
    return summary


def results_frame(results: List[BottomResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "date": result.date,
            "price": result.price,
            "square_price_time": result.square_price_time,
        }
        for target in PRICE_TARGETS:
            row[f"days_to_{target}"] = result.price_hits.get(target)
        for days in TIME_TARGETS:
            row[f"reversal_{days}d"] = result.time_reversals.get(days)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: str, ticker: str, summary: Dict[str, float], details: pd.DataFrame) -> None:
    summary_df = pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": list(summary.values()),
        }
    )
    summary_html = summary_df.to_html(index=False, float_format="{:.4f}".format)
    details_html = details.to_html(index=False)
    html = "\n".join(
        [
            f"<h1>Bayer-style verification report: {ticker}</h1>",
            "<h2>Summary</h2>",
            summary_html,
            "<h2>Bottom details</h2>",
            details_html,
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)


def format_summary(ticker: str, summary: Dict[str, float]) -> str:
    lines = [f"Results for {ticker}:", f"  bottoms analyzed: {summary.get('bottoms', 0)}"]
    for target in PRICE_TARGETS:
        lines.append(f"  price hit +{target}: {summary.get(f'price_hit_{target}', 0):.1%}")
    for days in TIME_TARGETS:
        lines.append(f"  time reversal at {days}d: {summary.get(f'time_reversal_{days}', 0):.1%}")
    lines.append(f"  square of price/time (+35 in <=35d): {summary.get('square_price_time', 0):.1%}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Bayer-style price/time hypotheses using Yahoo Finance data")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=20, help="Window size for local minima detection")
    parser.add_argument("--forward-days", type=int, default=180, help="Forward calendar days to scan for targets")
    parser.add_argument("--report", default=None, help="Optional HTML report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    for ticker in args.tickers:
        df = download_prices(ticker, args.start, end)
        minima = find_local_minima(df, args.window)
        results = evaluate_bottoms(df, minima, args.forward_days)
        summary = summarize(results)
        print(format_summary(ticker, summary))
        if args.report:
            details = results_frame(results)
            write_report(args.report, ticker, summary, details)
            print(f"Report written to {args.report}")
        print("-")


if __name__ == "__main__":
    main()
