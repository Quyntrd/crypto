"""
ЛР2, вариант 13: комбинирование LFSR-генераторов с помощью псевдослучайного
прореживания (SG-генератор, shrinking generator).

Модель LFSR - как в методичке, формула (8):
    S_i(t+1) = S_{i+1}(t),            i = 0..n-2
    S_{n-1}(t+1) = sum_j a_j S_j(t) mod 2
    выход: s_t = S_0(t)
Многочлен f(x) = x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0, a_0 = 1.

SG-генератор: G1 даёт "элементарную" последовательность {a_t}, G2 - селектирующую {s_t}.
В выход попадает a_t только если s_t = 1 (при s_t = 0 бит a_t отбрасывается).
"""
from math import gcd
import numpy as np


# ----------------------------------------------------------------------------
# Многочлены над GF(2). Многочлен хранится как int: бит e = коэффициент при x^e.
# ----------------------------------------------------------------------------
def poly_from_exps(exps):
    p = 0
    for e in exps:
        p |= 1 << e
    return p


def poly_str(p):
    terms = []
    for e in range(p.bit_length() - 1, -1, -1):
        if (p >> e) & 1:
            terms.append("1" if e == 0 else ("x" if e == 1 else f"x^{e}"))
    return " + ".join(terms)


def _mulmod(a, b, f):
    """a*b mod f над GF(2)."""
    n = f.bit_length() - 1
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if (a >> n) & 1:
            a ^= f
    return r


def _powmod(a, e, f):
    r = 1
    while e:
        if e & 1:
            r = _mulmod(r, a, f)
        a = _mulmod(a, a, f)
        e >>= 1
    return r


def _prime_factors(m):
    res, d = [], 2
    while d * d <= m:
        if m % d == 0:
            res.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        res.append(m)
    return res


def is_primitive(f):
    """
    Многочлен f степени n примитивен над GF(2) <=> порядок x по модулю f равен 2^n - 1,
    т.е. x^(2^n-1) = 1 и x^((2^n-1)/q) != 1 для каждого простого q | 2^n-1.
    (Из этого автоматически следует неприводимость.)
    """
    n = f.bit_length() - 1
    if n < 1 or not (f & 1):          # свободный член должен быть 1
        return False
    if n == 1:
        return f == 0b11
    order = (1 << n) - 1
    x = 0b10
    if _powmod(x, order, f) != 1:
        return False
    return all(_powmod(x, order // q, f) != 1 for q in _prime_factors(order))


def is_irreducible(f):
    """Проверка неприводимости перебором делителей (степени <= n/2); для малых n."""
    n = f.bit_length() - 1
    if n < 1:
        return False
    for d in range(2, 1 << (n // 2 + 1)):
        # остаток f mod d
        r, dd = f, d.bit_length() - 1
        for s in range(r.bit_length() - 1, dd - 1, -1):
            if (r >> s) & 1:
                r ^= d << (s - dd)
        if r == 0 and d != f:
            return False
    return True


def find_primitive(n):
    """Примитивный многочлен степени n с наименьшим числом слагаемых (затем - наименьший по значению)."""
    best = None
    for mid in range(1 << (n - 1)):
        f = (1 << n) | (mid << 1) | 1
        if best is not None and bin(f).count("1") > bin(best).count("1"):
            continue
        if is_primitive(f):
            if best is None or bin(f).count("1") < bin(best).count("1"):
                best = f
    return best


# ----------------------------------------------------------------------------
# LFSR
# ----------------------------------------------------------------------------
class LFSR:
    def __init__(self, f, seed):
        self.f = f
        self.n = f.bit_length() - 1
        self.taps = f & ((1 << self.n) - 1)       # a_j = коэффициент при x^j, j<n
        if seed <= 0 or seed >= (1 << self.n):
            raise ValueError("начальное состояние должно быть ненулевым n-битным числом")
        self.seed = seed
        self.state = seed

    def step(self):
        """Один такт: возвращает s_t = S_0(t) и сдвигает регистр по формуле (8)."""
        out = self.state & 1
        fb = bin(self.state & self.taps).count("1") & 1
        self.state = (self.state >> 1) | (fb << (self.n - 1))
        return out

    def cycle(self):
        """Выходная последовательность за полный цикл состояний (до возврата в seed)."""
        st = LFSR(self.f, self.seed)
        out = [st.step()]
        while st.state != st.seed:
            out.append(st.step())
        return np.array(out, dtype=np.uint8)

    def state_period(self):
        return len(self.cycle())


def min_period(seq):
    """Минимальный период чисто периодической последовательности seq (период делит len(seq))."""
    L = len(seq)
    for p in _divisors(L):
        if np.array_equal(seq[:L - p], seq[p:]):
            return p
    return L


def _divisors(m):
    small, large, d = [], [], 1
    while d * d <= m:
        if m % d == 0:
            small.append(d)
            if d != m // d:
                large.append(m // d)
        d += 1
    return small + large[::-1]


# ----------------------------------------------------------------------------
# SG-генератор
# ----------------------------------------------------------------------------
class ShrinkingGenerator:
    def __init__(self, f_data, seed_data, f_sel, seed_sel):
        self.g1 = LFSR(f_data, seed_data)
        self.g2 = LFSR(f_sel, seed_sel)
        self._a = self.g1.cycle()                  # период {a_t} (по циклу состояний)
        self._s = self.g2.cycle()                  # период {s_t}
        self.Ta, self.Ts = len(self._a), len(self._s)
        self.ones_per_Ts = int(self._s.sum())

    # --- эталонная побитовая реализация (медленная) -------------------------
    def bits_reference(self, length):
        g1, g2 = LFSR(self.g1.f, self.g1.seed), LFSR(self.g2.f, self.g2.seed)
        out = []
        while len(out) < length:
            a, s = g1.step(), g2.step()
            if s == 1:
                out.append(a)
        return np.array(out, dtype=np.uint8)

    # --- быстрая векторизованная реализация --------------------------------
    def bits(self, length):
        """Первые `length` бит выхода (через периодические таблицы LFSR)."""
        out = np.empty(0, dtype=np.uint8)
        clocks_done, chunk = 0, max(1000, int(2.2 * length) + 1000)
        parts, have = [], 0
        while have < length:
            t = np.arange(clocks_done, clocks_done + chunk, dtype=np.int64)
            a = self._a[t % self.Ta]
            s = self._s[t % self.Ts]
            sel = a[s == 1]
            parts.append(sel)
            have += len(sel)
            clocks_done += chunk
        return np.concatenate(parts)[:length]

    # --- период ------------------------------------------------------------
    def period_upper_bound(self):
        """K = число выходных бит за lcm(Ta,Ts) тактов; период выхода делит K."""
        L = self.Ta * self.Ts // gcd(self.Ta, self.Ts)
        return self.ones_per_Ts * (L // self.Ts)

    def period(self):
        K = self.period_upper_bound()
        x = self.bits(2 * K)
        return min_period(x[:K]) if np.array_equal(x[:K], x[K:2 * K]) else None

    @staticmethod
    def theoretical_period(n, m):
        """T = (2^n - 1) * 2^(m-1), если многочлены примитивны и gcd(Ta, Ts) = 1."""
        return ((1 << n) - 1) * (1 << (m - 1))
