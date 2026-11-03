"""
ЛР2, вариант 13: комбинирование с прореживающим (SG) генератором.
Тесты: спектральный, совпадение перекрывающихся шаблонов, линейная сложность.
Файлы data.e и data.pi должны лежать рядом с main.py.

Режим ref=True в тестах воспроизводит константы референсного кода NIST STS, по которым посчитана
Таблица 1 методички (старые pi_i для шаблонов, pi_0=0.01047 для линейной сложности).
Режим ref=False - теоретически точные константы (для справки).
"""
import math
from math import gcd
import numpy as np
from scipy.special import gammaincc

ALPHA = 0.01

# ----------------------------- LFSR и SG -----------------------------

def parity(x):
    return bin(x).count("1") & 1


class LFSR:
    """LFSR по модели (8) методички. taps - маска коэффициентов c_i многочлена
    f(x)=x^n + c_{n-1}x^{n-1}+...+c_0 (c_0=1). Выход - S_0."""
    def __init__(self, n, taps, seed=1):
        self.n, self.taps = n, taps
        self.mask = (1 << n) - 1
        self.state = (seed & self.mask) or 1
        self.seed = self.state

    def step(self):
        out = self.state & 1
        fb = parity(self.state & self.taps)
        self.state = (self.state >> 1) | (fb << (self.n - 1))
        return out

    def cycle(self):
        """Выход за полный цикл состояний (длина = период последовательности)."""
        g = LFSR(self.n, self.taps, self.seed)
        out = [g.step()]
        while g.state != g.seed:
            out.append(g.step())
        return np.array(out, dtype=np.uint8)


def poly_str(n, taps):
    t = ["x^%d" % n] + [("x^%d" % i if i > 1 else "x") for i in range(n - 1, 0, -1) if (taps >> i) & 1] + ["1"]
    return "+".join(t)


def is_primitive_poly(n, taps):
    """Примитивен <=> LFSR имеет максимальный период 2^n-1."""
    if (taps & 1) == 0:
        return False
    period = (1 << n) - 1
    state = 1
    for k in range(1, period + 1):
        fb = parity(state & taps)
        state = (state >> 1) | (fb << (n - 1))
        if state == 1:
            return k == period
    return False


def find_primitive(n, start=None):
    if start is not None and is_primitive_poly(n, start):
        return start
    for mask in range(1, 1 << n, 2):
        if is_primitive_poly(n, mask):
            return mask
    raise RuntimeError("Примитивный многочлен не найден")


class SG:
    """Shrinking generator: G1 - данные a_t, G2 - селектор s_t; выход - a_t при s_t = 1."""
    def __init__(self, n1, taps1, n2, taps2, seed1=1, seed2=1):
        self.g1, self.g2 = LFSR(n1, taps1, seed1), LFSR(n2, taps2, seed2)
        self.a, self.s = self.g1.cycle(), self.g2.cycle()
        self.Ta, self.Ts = len(self.a), len(self.s)

    def bits(self, num):
        """Первые num бит выхода (векторизованно, через периодические таблицы LFSR)."""
        parts, have, done, chunk = [], 0, 0, max(1000, int(2.2 * num) + 1000)
        while have < num:
            t = np.arange(done, done + chunk, dtype=np.int64)
            sel = self.a[t % self.Ta][self.s[t % self.Ts] == 1]
            parts.append(sel); have += len(sel); done += chunk
        return np.concatenate(parts)[:num]

    def bits_slow(self, num):
        """Эталонная побитовая реализация (для самопроверки)."""
        g1, g2 = LFSR(self.g1.n, self.g1.taps, self.g1.seed), LFSR(self.g2.n, self.g2.taps, self.g2.seed)
        out = []
        while len(out) < num:
            a, s = g1.step(), g2.step()
            if s:
                out.append(a)
        return np.array(out, dtype=np.uint8)

    def period(self):
        """Период выхода ПО МОДЕЛИРОВАНИЮ. За lcm(Ta,Ts) тактов система возвращается в исходное состояние,
        при этом выводится K бит -> период выхода делит K; берём минимальный делитель-период."""
        L = self.Ta * self.Ts // gcd(self.Ta, self.Ts)
        K = int(self.s.sum()) * (L // self.Ts)
        x = self.bits(2 * K)
        assert np.array_equal(x[:K], x[K:]), "K не является периодом"
        for p in divisors(K):
            if np.array_equal(x[:K - p], x[p:K]):
                return p, K


def divisors(m):
    lo, hi, d = [], [], 1
    while d * d <= m:
        if m % d == 0:
            lo.append(d)
            if d != m // d:
                hi.append(m // d)
        d += 1
    return lo + hi[::-1]


def sg_period_formula(n1, n2):
    """T=(2^n1-1)*2^(n2-1), если периоды регистров взаимно просты."""
    Ta, Ts = (1 << n1) - 1, (1 << n2) - 1
    if gcd(Ta, Ts) != 1:
        raise ValueError("Периоды LFSR не взаимно просты")
    return Ta * (1 << (n2 - 1))


# ----------------------------- Статистические тесты -----------------------------

def spectral_test(bits):
    """NIST 2.6, знаменатель sqrt(n*0.95*0.05/4)."""
    n = len(bits)
    X = 2.0 * np.asarray(bits, dtype=float) - 1.0
    M = np.abs(np.fft.fft(X))[: n // 2]
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = int(np.sum(M < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return math.erfc(abs(d) / math.sqrt(2.0)), d, N1, N0


def _pi_poisson(K=5, eta=1.0):
    """pi_i как в референсном коде STS (функция Pr; lambda=2, eta=1)."""
    def pr(u):
        if u == 0:
            return math.exp(-eta)
        return sum(math.exp(-eta - u * math.log(2) + l * math.log(eta) - math.lgamma(l + 1)
                            + math.lgamma(u) - math.lgamma(l) - math.lgamma(u - l + 1))
                   for l in range(1, u + 1))
    p = [pr(u) for u in range(K)]
    return p + [1.0 - sum(p)]


PI_OT_REF = _pi_poisson()                                             # эталон Таблицы 1
PI_OT_EXACT = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]   # Hamano-Kaneko


def overlapping_template_test(bits, m=9, M=1032, N=968, K=5, ref=True):
    """NIST 2.8. Шаблон - m единиц. Блоки по M бит, окно сдвигается на 1 бит."""
    pi = PI_OT_REF if ref else PI_OT_EXACT
    b = np.asarray(bits, dtype=np.uint8)
    v = [0] * (K + 1)
    for j in range(N):
        blk = b[j * M:(j + 1) * M]
        match = np.ones(M - m + 1, dtype=bool)
        for k in range(m):
            match &= blk[k:M - m + 1 + k] == 1
        v[min(int(match.sum()), K)] += 1
    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return float(gammaincc(K / 2.0, chi2 / 2.0)), chi2, v


def berlekamp_massey(block):
    """Линейная сложность блока (Берлекэмп-Мэсси над GF(2)); многочлены - целые числа-битмаски."""
    C, B, L, m, R = 1, 1, 0, -1, 0       # R: бит j = s[n-1-j]
    for n, sn in enumerate(block):
        sn = int(sn)
        d = sn ^ (((C >> 1) & R).bit_count() & 1)
        if d:
            T = C
            C ^= B << (n - m)
            if 2 * L <= n:
                L, B, m = n + 1 - L, T, n
        R = (R << 1) | sn
    return L


PI_LC_REF = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
PI_LC_EXACT = [1 / 96, 1 / 32, 1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 48]


def linear_complexity_test(bits, M=500, K=6, ref=True):
    """NIST 2.10."""
    pi = PI_LC_REF if ref else PI_LC_EXACT
    b = np.asarray(bits, dtype=np.uint8)
    N = len(b) // M
    mu = M / 2.0 + (9 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / 2.0 ** M
    v = [0] * (K + 1)
    for j in range(N):
        L = berlekamp_massey(b[j * M:(j + 1) * M].tolist())
        T = (-1) ** M * (L - mu) + 2.0 / 9.0
        v[0 if T <= -2.5 else 1 if T <= -1.5 else 2 if T <= -0.5 else
          3 if T <= 0.5 else 4 if T <= 1.5 else 5 if T <= 2.5 else 6] += 1
    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return float(gammaincc(K / 2.0, chi2 / 2.0)), chi2, v


def read_bits(path, n=1_000_000):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return (raw[(raw == 48) | (raw == 49)] - 48)[:n]


def verdict(p):
    return "PASS" if p >= ALPHA else "FAIL"


def run_tests(name, bits, expected=None):
    print(f"\n=== {name} ===")
    p, d, N1, N0 = spectral_test(bits)
    print("Спектральный:             P=%.6f (d=%.4f, N1=%d, N0=%.1f) -> %s" % (p, d, N1, N0, verdict(p)))
    ps = [p]
    p, chi2, v = overlapping_template_test(bits, ref=True)
    pe = overlapping_template_test(bits, ref=False)[0]
    print("Перекрыв. шаблоны:        P=%.6f (chi2=%.4f, v=%s) -> %s   [точные pi_i: P=%.6f]" % (p, chi2, v, verdict(p), pe))
    ps.append(p)
    p, chi2, v = linear_complexity_test(bits, ref=True)
    pe = linear_complexity_test(bits, ref=False)[0]
    print("Линейная сложность:       P=%.6f (chi2=%.4f, v=%s) -> %s   [точный pi_0: P=%.6f]" % (p, chi2, v, verdict(p), pe))
    ps.append(p)
    if expected:
        for nm, got, exp in zip(("Спектральный", "Шаблоны", "Лин. сложность"), ps, expected):
            print("   эталон %-15s %.6f  получено %.6f  %s" % (nm, exp, got, "СОВПАДАЕТ" if abs(got - exp) < 5e-7 else "РАСХОДИТСЯ"))
    return ps


# ----------------------------- main -----------------------------

if __name__ == "__main__":
    TABLE_COL_PI = (0.847187, 0.110434, 0.826335)   # получаются на data.e
    TABLE_COL_E = (0.010186, 0.296897, 0.255475)    # получаются на data.pi

    # 1. Примитивность многочленов варианта 13
    variants = [(7, 0b1111, 1), (7, 0b0011, 2), (7, 0b1001, 3)]
    print("Проверка примитивности многочленов варианта 13:")
    for n, taps, k in variants:
        print("  случай %d: %-20s -> %s" % (k, poly_str(n, taps), "primitive" if is_primitive_poly(n, taps) else "NOT primitive"))

    # 2. Проверка работоспособности тестов на data.e и data.pi
    run_tests("data.e  (в Таблице 1 - столбец 'π')", read_bits("data.e"), TABLE_COL_PI)
    run_tests("data.pi (в Таблице 1 - столбец 'e')", read_bits("data.pi"), TABLE_COL_E)

    # 3. Подбор параметров SG: G1 - многочлен варианта (n1=7), селектор G2 - n2=15
    n2 = 15
    taps2 = find_primitive(n2, 0b11)               # x^15 + x + 1
    print("\nСелектор G2: %s, примитивен: %s" % (poly_str(n2, taps2), is_primitive_poly(n2, taps2)))
    print("gcd(2^7-1, 2^%d-1) = %d; два регистра степени 7 дали бы период 64 (gcd=127)" % (n2, gcd(127, (1 << n2) - 1)))
    print("Период по формуле: %d" % sg_period_formula(7, n2))

    # 4. Три случая: генерация, период, тесты
    for n1, taps1, k in variants:
        sg = SG(n1, taps1, n2, taps2, seed1=0b1011001, seed2=12345)
        assert np.array_equal(sg.bits(5000), sg.bits_slow(5000))     # самопроверка быстрой реализации
        T, K = sg.period()
        print("\n############ Случай %d: G1 = %s ############" % (k, poly_str(n1, taps1)))
        print("Период по моделированию: %d (формула: %d) -> %s" % (T, sg_period_formula(n1, n2), "совпал" if T == sg_period_formula(n1, n2) else "НЕ совпал"))
        bits = sg.bits(1_000_000)
        with open("seq_case%d.txt" % k, "w") as f:
            f.write("".join(map(str, bits.tolist())))
        print("Сгенерировано бит: %d, доля единиц: %.5f" % (len(bits), bits.mean()))
        run_tests("SG, случай %d" % k, bits)