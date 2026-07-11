def h_score(asr: float, sro: float) -> float:
    denom = asr + (1.0 - sro)
    if abs(denom) < 1e-12:
        return float("nan")
    return 2.0 * asr * (1.0 - sro) / denom


raw_scores = """
&0.88&0.4091&
"""

for line in raw_scores.strip().splitlines():
    parts = [p.strip() for p in line.split("&") if p.strip()]
    if len(parts) < 2:
        continue

    asr = float(parts[0])
    sro = float(parts[1])
    h = h_score(asr, sro)

    print(f"&{asr:.2f}&{sro:.4f}&{h:.4f}")
