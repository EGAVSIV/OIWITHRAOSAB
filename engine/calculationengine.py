"""
engine/scanner.py — Calculation Engine (no UI)

Runs on a schedule (GitHub Actions, every ~5 min during NSE market hours).
Fetches the option chain from Upstox for every symbol in the master file,
then writes three JSON outputs consumed by the static dashboard in /docs:

  docs/data/scan.json              -> Tab 1: OI decay Call/Put signal scanner
  docs/data/chain_latest.json      -> Tab 3: full option chain snapshot w/ greeks
  docs/data/history/<date>/<SYM>.json -> Tab 2: intraday time-series per symbol
                                          (OI, Gamma, Vega, Theta, IV, RV)

The Upstox access token is read from the UPSTOX_TOKEN environment variable
(set as a GitHub Actions secret). It never leaves this script — only the
computed JSON (no token, no credentials) is written to /docs for the
public dashboard to read.
"""

import gzip
import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

# ---------------------------- CONFIG ----------------------------
IST = ZoneInfo("Asia/Kolkata")
BASE_URL = "https://api.upstox.com/v2"

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_FILE = REPO_ROOT / "engine" / "complete.json.gz"
DATA_DIR = REPO_ROOT / "docs" / "data"
HISTORY_DIR = DATA_DIR / "history"

DECAY_LIMIT = -5.0          # % — same condition as the original scanner
HISTORY_STRIKE_WINDOW = 3   # keep ATM +/- N strikes in intraday history (file-size control)
RISK_FREE_RATE = 0.07       # used only if API doesn't supply greeks directly

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


# ---------------------------- AUTH ----------------------------
def load_access_token():
    token = os.environ.get("UPSTOX_TOKEN", "").strip()
    if token:
        return token
    # local dev fallback
    local = REPO_ROOT / "engine" / "token.txt"
    if local.exists():
        return local.read_text().strip()
    raise SystemExit("No Upstox token found (set UPSTOX_TOKEN env var or engine/token.txt)")


ACCESS_TOKEN = load_access_token()
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}


# ---------------------------- MARKET HOURS GUARD ----------------------------
def market_is_open(now_ist=None):
    now_ist = now_ist or datetime.now(IST)
    if now_ist.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now_ist.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_t = now_ist.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return open_t <= now_ist <= close_t


# ---------------------------- MASTER / SYMBOLS ----------------------------
def load_master():
    with gzip.open(MASTER_FILE, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_symbol_map(master):
    sym_to_inst = {}
    for x in master:
        sy = x.get("underlying_symbol")
        uk = x.get("underlying_key")
        if sy and uk and sy not in sym_to_inst:
            sym_to_inst[sy] = uk
    return sym_to_inst


# ---------------------------- EXPIRY / CHAIN FETCH ----------------------------
def safe_expiry(raw):
    try:
        if isinstance(raw, str):
            return raw[:10]
        if raw > 1e12:
            return datetime.utcfromtimestamp(raw / 1000).strftime("%Y-%m-%d")
        return datetime.utcfromtimestamp(raw).strftime("%Y-%m-%d")
    except Exception:
        return None


def get_expiries(instrument_key):
    r = requests.get(f"{BASE_URL}/option/contract", headers=HEADERS,
                      params={"instrument_key": instrument_key}, timeout=15)
    if r.status_code != 200:
        return []
    out = [safe_expiry(d.get("expiry")) for d in r.json().get("data", [])]
    return sorted({e for e in out if e})


def get_chain(inst, expiry):
    r = requests.get(f"{BASE_URL}/option/chain", headers=HEADERS,
                      params={"instrument_key": inst, "expiry_date": expiry}, timeout=20)
    if r.status_code != 200:
        return []
    return r.json().get("data", [])


# ---------------------------- GREEKS (fallback Black-Scholes) ----------------------------
def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_greeks(spot, strike, t_years, iv, option_type):
    """Fallback Greeks if Upstox doesn't supply option_greeks directly."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return {"gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sigma = iv / 100.0
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    gamma = _norm_pdf(d1) / (spot * sigma * math.sqrt(t_years))
    vega = spot * _norm_pdf(d1) * math.sqrt(t_years) / 100  # per 1% IV move
    if option_type == "CE":
        theta = (-(spot * _norm_pdf(d1) * sigma) / (2 * math.sqrt(t_years))
                 - RISK_FREE_RATE * strike * math.exp(-RISK_FREE_RATE * t_years) * _norm_cdf(d2)) / 365
    else:
        theta = (-(spot * _norm_pdf(d1) * sigma) / (2 * math.sqrt(t_years))
                 + RISK_FREE_RATE * strike * math.exp(-RISK_FREE_RATE * t_years) * _norm_cdf(-d2)) / 365
    return {"gamma": round(gamma, 6), "vega": round(vega, 4), "theta": round(theta, 4)}


def years_to_expiry(expiry_str, now_ist):
    expiry_close = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
        hour=15, minute=30, tzinfo=IST)
    delta = (expiry_close - now_ist).total_seconds()
    return max(delta, 60) / (365 * 24 * 3600)


def extract_leg(leg, spot, strike, t_years, option_type):
    md = leg.get("market_data", {}) or {}
    og = leg.get("option_greeks", {}) or {}

    oi = md.get("oi", 0) or 0
    prev_oi = md.get("prev_oi", 0) or 0
    ltp = md.get("ltp", 0) or 0
    iv = og.get("iv", md.get("iv", 0)) or 0

    gamma = og.get("gamma")
    vega = og.get("vega")
    theta = og.get("theta")

    if gamma is None or vega is None or theta is None:
        computed = bs_greeks(spot, strike, t_years, iv, option_type)
        gamma = gamma if gamma is not None else computed["gamma"]
        vega = vega if vega is not None else computed["vega"]
        theta = theta if theta is not None else computed["theta"]

    decay = 0.0 if prev_oi == 0 else ((oi - prev_oi) / prev_oi) * 100

    return {
        "oi": oi, "prev_oi": prev_oi, "decay": round(decay, 2),
        "ltp": ltp, "iv": round(float(iv), 2),
        "gamma": round(float(gamma), 6), "vega": round(float(vega), 4),
        "theta": round(float(theta), 4),
    }


# ---------------------------- MAIN PROCESSING ----------------------------
def process_symbol(sym, inst, now_ist):
    expiries = get_expiries(inst)
    if not expiries:
        return None
    expiry = expiries[0]
    raw = get_chain(inst, expiry)
    if not raw:
        return None

    spot = float(raw[0].get("underlying_spot_price", 0) or 0)
    t_years = years_to_expiry(expiry, now_ist)

    rows = []
    for x in raw:
        strike = x.get("strike_price", 0)
        ce = extract_leg(x.get("call_options", {}) or {}, spot, strike, t_years, "CE")
        pe = extract_leg(x.get("put_options", {}) or {}, spot, strike, t_years, "PE")
        rows.append({"strike": strike, "ce": ce, "pe": pe})

    rows.sort(key=lambda r: r["strike"])
    return {"symbol": sym, "spot": round(spot, 2), "expiry": expiry, "rows": rows}


def compute_scan_signal(chain):
    strikes = [r["strike"] for r in chain["rows"]]
    spot = chain["spot"]
    if len(strikes) < 5:
        return None

    atm_strike = min(strikes, key=lambda s: abs(s - spot))
    atm_idx = strikes.index(atm_strike)
    if atm_idx < 2 or atm_idx > len(strikes) - 3:
        return None

    by_strike = {r["strike"]: r for r in chain["rows"]}
    ce_atm = by_strike[strikes[atm_idx]]["ce"]
    ce_o1 = by_strike[strikes[atm_idx + 1]]["ce"]
    ce_o2 = by_strike[strikes[atm_idx + 2]]["ce"]
    pe_atm = by_strike[strikes[atm_idx]]["pe"]
    pe_o1 = by_strike[strikes[atm_idx - 1]]["pe"]
    pe_o2 = by_strike[strikes[atm_idx - 2]]["pe"]

    call_signal = (ce_atm["decay"] <= DECAY_LIMIT and ce_o1["decay"] <= DECAY_LIMIT and
                   ce_o2["decay"] <= DECAY_LIMIT and pe_atm["decay"] > 0 and pe_o1["decay"] > 0)
    put_signal = (pe_atm["decay"] <= DECAY_LIMIT and pe_o1["decay"] <= DECAY_LIMIT and
                  pe_o2["decay"] <= DECAY_LIMIT and ce_atm["decay"] > 0 and ce_o1["decay"] > 0
                  and ce_o2["decay"] > 0)

    if not (call_signal or put_signal):
        return None

    signal = []
    if call_signal:
        signal.append("CALL")
    if put_signal:
        signal.append("PUT")

    return {
        "symbol": chain["symbol"], "close": spot, "signal": "/".join(signal),
        "atm_strike": strikes[atm_idx],
        "ce_atm_dec": ce_atm["decay"], "ce_otm1_dec": ce_o1["decay"], "ce_otm2_dec": ce_o2["decay"],
        "pe_atm_dec": pe_atm["decay"], "pe_otm1_dec": pe_o1["decay"], "pe_otm2_dec": pe_o2["decay"],
    }


def append_history(sym, chain, now_ist):
    """Append an intraday snapshot (ATM +/- window) for charting, reset daily."""
    date_str = now_ist.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    fpath = day_dir / f"{sym}.json"

    existing = {"symbol": sym, "points": []}
    if fpath.exists():
        try:
            existing = json.loads(fpath.read_text())
        except Exception:
            pass

    strikes = [r["strike"] for r in chain["rows"]]
    spot = chain["spot"]
    atm_strike = min(strikes, key=lambda s: abs(s - spot))
    atm_idx = strikes.index(atm_strike)
    lo = max(0, atm_idx - HISTORY_STRIKE_WINDOW)
    hi = min(len(strikes), atm_idx + HISTORY_STRIKE_WINDOW + 1)
    window_rows = chain["rows"][lo:hi]

    # realized-volatility proxy from today's spot prints so far
    prev_spots = [p["spot"] for p in existing["points"]]
    rv = None
    if len(prev_spots) >= 2:
        rets = [math.log(prev_spots[i] / prev_spots[i - 1]) for i in range(1, len(prev_spots))
                if prev_spots[i - 1] > 0]
        if rets:
            mean = sum(rets) / len(rets)
            var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
            # annualize assuming 5-min bars, ~75 bars/session, 252 sessions/yr
            rv = round(math.sqrt(var) * math.sqrt(75 * 252) * 100, 2)

    point = {
        "t": now_ist.strftime("%H:%M"),
        "spot": spot,
        "rv": rv,
        "atm_strike": atm_strike,
        "strikes": [
            {
                "strike": r["strike"],
                "ce_oi": r["ce"]["oi"], "pe_oi": r["pe"]["oi"],
                "ce_iv": r["ce"]["iv"], "pe_iv": r["pe"]["iv"],
                "ce_gamma": r["ce"]["gamma"], "pe_gamma": r["pe"]["gamma"],
                "ce_vega": r["ce"]["vega"], "pe_vega": r["pe"]["vega"],
                "ce_theta": r["ce"]["theta"], "pe_theta": r["pe"]["theta"],
            }
            for r in window_rows
        ],
    }
    existing["points"].append(point)
    fpath.write_text(json.dumps(existing))


def best_strikes_for_buying(chain, top_n=3):
    """Simple heuristic ranking: reward gamma & OI (liquidity), penalise theta bleed & rich IV."""
    def score(leg):
        gamma = leg["gamma"] or 0
        theta = abs(leg["theta"] or 0.0001)
        oi = leg["oi"] or 0
        iv = leg["iv"] or 1
        return (gamma * math.log10(oi + 10)) / (theta * math.sqrt(iv))

    ce_ranked = sorted(chain["rows"], key=lambda r: score(r["ce"]), reverse=True)[:top_n]
    pe_ranked = sorted(chain["rows"], key=lambda r: score(r["pe"]), reverse=True)[:top_n]
    return {
        "ce": [{"strike": r["strike"], **r["ce"]} for r in ce_ranked],
        "pe": [{"strike": r["strike"], **r["pe"]} for r in pe_ranked],
    }


def main():
    now_ist = datetime.now(IST)
    if not market_is_open(now_ist) and os.environ.get("FORCE_RUN") != "1":
        print("Market closed — skipping run.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master = load_master()
    sym_to_inst = build_symbol_map(master)
    symbols = sorted(sym_to_inst.keys())

    scan_rows = []
    chain_snapshot = {}

    for sym in symbols:
        inst = sym_to_inst[sym]
        try:
            chain = process_symbol(sym, inst, now_ist)
        except Exception as e:
            print(f"[{sym}] error: {e}")
            continue
        if not chain:
            continue

        sig = compute_scan_signal(chain)
        if sig:
            scan_rows.append(sig)

        chain_snapshot[sym] = {
            "symbol": sym, "spot": chain["spot"], "expiry": chain["expiry"],
            "updated_at": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": [
                {"strike": r["strike"],
                 "ce_oi": r["ce"]["oi"], "ce_decay": r["ce"]["decay"], "ce_iv": r["ce"]["iv"],
                 "ce_gamma": r["ce"]["gamma"], "ce_vega": r["ce"]["vega"], "ce_theta": r["ce"]["theta"],
                 "ce_ltp": r["ce"]["ltp"],
                 "pe_oi": r["pe"]["oi"], "pe_decay": r["pe"]["decay"], "pe_iv": r["pe"]["iv"],
                 "pe_gamma": r["pe"]["gamma"], "pe_vega": r["pe"]["vega"], "pe_theta": r["pe"]["theta"],
                 "pe_ltp": r["pe"]["ltp"]}
                for r in chain["rows"]
            ],
            "best_strikes": best_strikes_for_buying(chain),
        }

        append_history(sym, chain, now_ist)
        time.sleep(0.05)  # be gentle on the API

    (DATA_DIR / "scan.json").write_text(json.dumps({
        "updated_at": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
        "decay_threshold": DECAY_LIMIT,
        "rows": scan_rows,
    }))

    (DATA_DIR / "chain_latest.json").write_text(json.dumps({
        "updated_at": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": chain_snapshot,
    }))

    (DATA_DIR / "symbols.json").write_text(json.dumps(sorted(chain_snapshot.keys())))

    print(f"Done. {len(scan_rows)} signals, {len(chain_snapshot)} symbols processed.")


if __name__ == "__main__":
    main()
