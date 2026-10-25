"""Полный прогон ЛР2, вариант 13: проверка многочленов, подбор параметров, период, тесты."""
import json, time
from math import gcd
import numpy as np
from prng import *
from stat_tests import *

UP = "/mnt/user-data/uploads/"
CASES = {1: [7, 3, 2, 1, 0], 2: [7, 1, 0], 3: [7, 3, 0]}
R = {}

# 1. примитивность
R["poly"] = {k: dict(poly=poly_str(poly_from_exps(e)), primitive=is_primitive(poly_from_exps(e)),
                     period=LFSR(poly_from_exps(e), 1).state_period()) for k, e in CASES.items()}

# 2. подбор степени селектора m (G1 = x^7+x+1)
f1 = poly_from_exps(CASES[2]); rows = []
for m in range(3, 16):
    f2 = find_primitive(m); g = ShrinkingGenerator(f1, 89, f2, 1 + (12345 % ((1 << m) - 1)))
    rows.append(dict(m=m, poly=poly_str(f2), gcd=gcd(g.Ta, g.Ts), T=g.period(), formula=g.theoretical_period(7, m)))
    print(rows[-1], flush=True)
R["m_search"] = rows

# 3. пары степени 7 (почему нельзя брать два регистра степени 7)
pairs = []
for i in CASES:
    for j in CASES:
        g = ShrinkingGenerator(poly_from_exps(CASES[i]), 89, poly_from_exps(CASES[j]), 45)
        pairs.append(dict(g1=i, g2=j, T=g.period()))
R["pairs7"] = pairs; print(pairs)

# 4. проверка тестов на pi и e (Таблица 1, столбцы переставлены)
val = {}
for nm in ("pi", "e"):
    x = read_bits(UP + "data." + nm, 10**6)
    val[nm] = dict(spectral=spectral_test(x)[0], overlap=overlapping_template_test(x)[0],
                   overlap_corrected=overlapping_template_test(x, pi=PI_OVERLAP_CORRECTED)[0],
                   lc=linear_complexity_test(x, 500, pi=PI_LINEAR_STS)[0], lc_exact=linear_complexity_test(x, 500)[0])
R["validation"] = val; print(val)

# 5. генерация 1e6 бит и тесты
f2 = poly_from_exps([15, 1, 0]); res = {}
def run(bits):
    return dict(spectral=spectral_test(bits)[0], overlap=overlapping_template_test(bits)[0],
                lc=linear_complexity_test(bits, 500)[0])
for k, e in CASES.items():
    g = ShrinkingGenerator(poly_from_exps(e), 89, f2, 12345)
    b = g.bits(10**6)
    open(f"seq_case{k}.txt", "w").write("".join(map(str, b.tolist())))
    res[k] = run(b); res[k]["ones"] = int(b.sum()); res[k]["period"] = g.period()
    print(k, res[k], flush=True)
R["cases"] = res

# 6. негативные контроли
neg = {}
b = LFSR(f1, 89).cycle(); b = np.tile(b, 10**6 // len(b) + 1)[:10**6]; neg["LFSR x^7+x+1"] = run(b)
g = ShrinkingGenerator(poly_from_exps(CASES[2]), 89, poly_from_exps(CASES[3]), 45); neg["SG 7+7"] = run(g.bits(10**6))
R["negative"] = neg; print(neg)

# 7. устойчивость к начальным состояниям
rng = np.random.default_rng(1); rob = {}
for k, e in CASES.items():
    cnt = dict(spectral=0, overlap=0, lc=0); N = 20
    for _ in range(N):
        g = ShrinkingGenerator(poly_from_exps(e), int(rng.integers(1, 128)), f2, int(rng.integers(1, 32768)))
        r = run(g.bits(10**6))
        for t in cnt: cnt[t] += r[t] >= ALPHA
    rob[k] = dict(N=N, passed=cnt)
R["robust"] = rob; print(rob)
json.dump(R, open("results.json", "w"), ensure_ascii=False, indent=1)
