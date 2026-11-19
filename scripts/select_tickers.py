"""
Ticker Selection and Health Check Script
Selects 20 clean tickers spanning >= 4 GICS sectors from 2010-01-01 to 2024-12-31.
Excludes companies with:
- Missing dates (incomplete history)
- Placeholder constant prices (e.g., SW, FERG, AMCR)
- Zero-volume days
- High == Low days
Assigns:
- 12 tickers to Set A (for Part 1 vision models)
- 8 tickers to Set B (for Part 2 market basket)
"""

import csv
from collections import defaultdict

def main():
    # 1. Load companies and their GICS sectors
    company_info = {}
    with open('Data/sp500_companies.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_info[row['symbol']] = {
                'name': row['company'],
                'sector': row['sector'],
                'sub_industry': row['sub_industry']
            }

    print(f"Loaded {len(company_info)} companies from sp500_companies.csv")

    # 2. First pass: count trading dates across all tickers in 2010-2024 to find the expected trading calendar
    date_counts = defaultdict(int)
    ticker_stats = defaultdict(lambda: {
        'total_days': 0,
        'zero_vol': 0,
        'high_eq_low': 0,
        'min_date': '9999-99-99',
        'max_date': '0000-00-00',
        'dates': set(),
        'prices': []
    })

    start_date = '2010-01-01'
    end_date = '2024-12-31'

    print("Scanning sp500_stocks.csv for 2010-2024...")
    with open('Data/sp500_stocks.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['date']
            if d < start_date or d > end_date:
                continue
            
            sym = row['symbol']
            date_counts[d] += 1
            
            try:
                op = float(row['open'])
                hi = float(row['high'])
                lo = float(row['low'])
                cl = float(row['close'])
                vol = float(row['volume'])
            except (ValueError, TypeError):
                continue
            
            stats = ticker_stats[sym]
            stats['total_days'] += 1
            if d < stats['min_date']:
                stats['min_date'] = d
            if d > stats['max_date']:
                stats['max_date'] = d
            
            if vol <= 0:
                stats['zero_vol'] += 1
            if hi <= lo or abs(hi - lo) < 1e-6:
                stats['high_eq_low'] += 1
            
            stats['dates'].add(d)

    # Determine master calendar (dates shared by most tickers)
    all_dates = sorted(date_counts.keys())
    total_market_days = len(all_dates)
    print(f"Total market trading dates between {start_date} and {end_date}: {total_market_days}")

    # 3. Filter candidates
    valid_candidates = []
    for sym, stats in ticker_stats.items():
        if sym not in company_info:
            continue
        # Must have data across the entire range
        if stats['min_date'] > '2010-01-10' or stats['max_date'] < '2024-10-01':
            continue
        # Check date coverage (must match > 99.5% of market days)
        if len(stats['dates']) < total_market_days - 5:
            continue
        # Must have no zero volume days
        if stats['zero_vol'] > 0:
            continue
        # Must have no High == Low days
        if stats['high_eq_low'] > 0:
            continue
        
        valid_candidates.append({
            'symbol': sym,
            'name': company_info[sym]['name'],
            'sector': company_info[sym]['sector'],
            'total_days': stats['total_days']
        })

    print(f"Found {len(valid_candidates)} perfectly clean candidates.")

    # Group by sector
    by_sector = defaultdict(list)
    for c in valid_candidates:
        by_sector[c['sector']].append(c)

    print("\nCandidate count per sector:")
    for sec, items in sorted(by_sector.items(), key=lambda x: -len(x[1])):
        print(f"  {sec}: {len(items)}")

    # 4. Pick 20 diverse, highly liquid, household-name tickers across at least 4 sectors
    # E.g., Tech, Financials, Health Care, Industrials, Consumer Discretionary
    preferred_picks = {
        'Information Technology': ['AAPL', 'MSFT', 'NVDA', 'INTC', 'CSCO'],
        'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        'Health Care': ['JNJ', 'PFE', 'UNH', 'ABT'],
        'Industrials': ['CAT', 'HON', 'GE', 'UPS'],
        'Consumer Discretionary': ['AMZN', 'MCD', 'NKE', 'HD'],
        'Energy': ['XOM', 'CVX', 'SLB']
    }

    selected = []
    for sec, preferred in preferred_picks.items():
        candidates_in_sec = {c['symbol']: c for c in by_sector[sec]}
        for sym in preferred:
            if sym in candidates_in_sec and len(selected) < 20:
                selected.append(candidates_in_sec[sym])

    print(f"\nSelected {len(selected)} tickers across {len(set(s['sector'] for s in selected))} sectors:")
    for s in selected:
        print(f"  {s['symbol']} - {s['name']} ({s['sector']})")

    # Split into Set A (12 tickers) and Set B (8 tickers)
    # Balanced across sectors
    set_a = []
    set_b = []
    sector_counts_a = defaultdict(int)
    sector_counts_b = defaultdict(int)

    for i, s in enumerate(selected):
        if len(set_a) < 12 and (len(selected) - i <= 8 - len(set_b) or len(set_a) < 12 and i % 5 != 0):
            set_a.append(s)
            sector_counts_a[s['sector']] += 1
        else:
            set_b.append(s)
            sector_counts_b[s['sector']] += 1

    # Ensure exactly 12 in A and 8 in B
    if len(set_a) != 12:
        while len(set_a) > 12:
            set_b.append(set_a.pop())
        while len(set_a) < 12:
            set_a.append(set_b.pop())

    print("\n--- Set A (12 Tickers for Part 1 Vision Models) ---")
    for s in set_a:
        print(f"  {s['symbol']:<5} | {s['sector']:<24} | {s['name']}")

    print("\n--- Set B (8 Tickers for Part 2 Market Basket) ---")
    for s in set_b:
        print(f"  {s['symbol']:<5} | {s['sector']:<24} | {s['name']}")

    # Save to a config file for use by notebooks
    with open('results/selected_tickers.json', 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'set_a': set_a,
            'set_b': set_b,
            'total_market_days': total_market_days,
            'start_date': start_date,
            'end_date': end_date
        }, f, indent=2)
    print("\nSaved selected tickers to results/selected_tickers.json")

if __name__ == '__main__':
    main()
