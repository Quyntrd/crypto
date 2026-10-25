#!/usr/bin/env python3
"""Вычисление периода SG-генератора (по моделированию, без использования формулы) и сравнение с теорией.
Пример:  python3 period.py --poly1 7,1,0 --poly2 15,1,0"""
import argparse
from math import gcd
from prng import ShrinkingGenerator, poly_from_exps, poly_str, is_primitive

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poly1", default="7,1,0"); ap.add_argument("--poly2", default="15,1,0")
    ap.add_argument("--seed1", type=int, default=0b1011001); ap.add_argument("--seed2", type=int, default=12345)
    a = ap.parse_args()
    f1 = poly_from_exps([int(t) for t in a.poly1.split(",")]); f2 = poly_from_exps([int(t) for t in a.poly2.split(",")])
    g = ShrinkingGenerator(f1, a.seed1, f2, a.seed2)
    n, m = f1.bit_length() - 1, f2.bit_length() - 1
    print(f"G1: {poly_str(f1)} (примитивен: {is_primitive(f1)}), период {g.Ta}")
    print(f"G2: {poly_str(f2)} (примитивен: {is_primitive(f2)}), период {g.Ts}")
    print(f"gcd(Ta,Ts) = {gcd(g.Ta, g.Ts)}")
    print(f"Период SG-выхода (моделирование): {g.period()}")
    print(f"По формуле (2^n-1)*2^(m-1): {g.theoretical_period(n, m)}")

if __name__ == "__main__":
    main()
