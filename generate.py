#!/usr/bin/env python3
"""Генератор двоичной псевдослучайной последовательности (SG-генератор), вариант 13.
Пример:  python3 generate.py --case 1 --length 1000000 --out seq_case1.txt
Параметры: --case 1|2|3  (многочлен G1, степень 7), либо --poly1 "7,3,2,1,0" (показатели степеней);
           --poly2 "15,1,0" (селектор), --seed1, --seed2 (ненулевые начальные состояния)."""
import argparse, sys
from prng import ShrinkingGenerator, poly_from_exps, poly_str, is_primitive

CASES = {1: [7, 3, 2, 1, 0], 2: [7, 1, 0], 3: [7, 3, 0]}
SELECTOR_DEFAULT = [15, 1, 0]          # x^15 + x + 1, примитивный; gcd(2^7-1, 2^15-1) = 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, choices=CASES, default=1)
    ap.add_argument("--poly1", type=str, help="показатели степеней G1, напр. 7,1,0")
    ap.add_argument("--poly2", type=str, default=",".join(map(str, SELECTOR_DEFAULT)))
    ap.add_argument("--seed1", type=int, default=0b1011001)
    ap.add_argument("--seed2", type=int, default=12345)
    ap.add_argument("--length", type=int, default=10**6)
    ap.add_argument("--out", type=str, default="-")
    a = ap.parse_args()
    e1 = [int(t) for t in a.poly1.split(",")] if a.poly1 else CASES[a.case]
    e2 = [int(t) for t in a.poly2.split(",")]
    f1, f2 = poly_from_exps(e1), poly_from_exps(e2)
    for nm, f in (("G1", f1), ("G2", f2)):
        print(f"{nm}: {poly_str(f)}  примитивен: {is_primitive(f)}", file=sys.stderr)
    g = ShrinkingGenerator(f1, a.seed1, f2, a.seed2)
    s = "".join("1" if b else "0" for b in g.bits(a.length))
    if a.out == "-":
        print(s)
    else:
        open(a.out, "w").write(s)
        print(f"записано {a.length} бит в {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
