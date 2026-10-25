"""
Три статистических теста NIST STS для варианта 13:
  6. Спектральный (Discrete Fourier Transform)
  8. Совпадение перекрывающихся шаблонов (Overlapping Template Matching)
 10. На линейную сложность (Linear Complexity), алгоритм Берлекэмпа-Мэсси
Каждый тест возвращает P-value; последовательность считается случайной, если P-value >= alpha (alpha = 0.01).
"""
import math
import numpy as np
from scipy.special import gammaincc, erfc   # igamc / erfc из NIST (специальные функции)

ALPHA = 0.01


# --------------------------------------------------------------------------- 6
def spectral_test(bits, denom=4.0):
    """
    NIST 2.6.  X = 2*eps - 1; S = DFT(X); M = |S[0 : n/2]|;
    T = sqrt(ln(1/0.05) * n); N0 = 0.95*n/2; N1 = #{M < T};
    d = (N1 - N0) / sqrt(n*0.95*0.05/denom);  P = erfc(|d|/sqrt(2)).
    denom=4 - формула из текста NIST SP 800-22 rev.1a (2.6.4);
    denom=3.8 - поправка Kim-Umeno-Hasegawa (STS 2.1.2).
    """
    x = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    n = len(x)
    m = np.abs(np.fft.fft(x))[: n // 2]
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = int(np.count_nonzero(m < T))
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / denom)
    return float(erfc(abs(d) / math.sqrt(2.0))), {"N0": n0, "N1": n1, "d": d, "T": T}


# --------------------------------------------------------------------------- 8
# теоретические вероятности pi_0..pi_5 (K=5, M=1032, m=9): исправленные (Hamano-Kaneko), NIST 3.8
PI_OVERLAP_CORRECTED = [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865]


def pi_overlap_poisson(eta=1.0, K=5):
    """Вероятности pi_0..pi_K по формуле составного пуассоновского закона (как в коде STS 2.1.x,
    функция Pr): lambda = (M-m+1)/2^m = 2, eta = lambda/2 = 1.  Дают эталонные P-value из NIST/Таблицы 1."""
    def pr(u):
        if u == 0:
            return math.exp(-eta)
        return sum(math.exp(-eta - u * math.log(2) + l * math.log(eta) - math.lgamma(l + 1)
                            + math.lgamma(u) - math.lgamma(l) - math.lgamma(u - l + 1))
                   for l in range(1, u + 1))
    p = [pr(u) for u in range(K)]
    p.append(1.0 - sum(p))
    return p


PI_OVERLAP_OLD = pi_overlap_poisson()


def overlapping_template_test(bits, template=(1,) * 9, M=1032, K=5, pi=None):
    """
    NIST 2.8.  Разбиваем на N = n // M блоков по M бит, в каждом блоке считаем
    число (перекрывающихся) вхождений шаблона B длины m; v_i - число блоков с i вхождениями
    (последний класс - K и более); chi2 = sum (v_i - N pi_i)^2/(N pi_i); P = igamc(K/2, chi2/2).
    """
    pi = PI_OVERLAP_OLD if pi is None else pi
    b = np.asarray(bits, dtype=np.uint8)
    n, m = len(b), len(template)
    N = n // M
    tpl = np.asarray(template, dtype=np.uint8)
    v = [0] * (K + 1)
    for j in range(N):
        blk = b[j * M:(j + 1) * M]
        # окно длины m скользит на 1 бит -> все позиции 0..M-m
        match = np.ones(M - m + 1, dtype=bool)
        for k in range(m):
            match &= blk[k:M - m + 1 + k] == tpl[k]
        cnt = int(match.sum())
        v[min(cnt, K)] += 1
    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return float(gammaincc(K / 2.0, chi2 / 2.0)), {"v": v, "chi2": chi2, "N": N}


# --------------------------------------------------------------------------- 10
def berlekamp_massey(block):
    """Линейная сложность L блока бит (алгоритм Берлекэмпа-Мэсси над GF(2)).
    Многочлены связи хранятся как целые числа (бит i - коэффициент при x^i), что даёт O(1) операций над
    длинными целыми на шаг."""
    C, B = 1, 1
    L, m = 0, -1
    R = 0                                    # бит j = s[n-1-j] (регистр прошлых бит)
    for n, sn in enumerate(block):
        d = int(sn) ^ (((C >> 1) & R).bit_count() & 1)
        if d:
            T = C
            C ^= B << (n - m)
            if 2 * L <= n:
                L = n + 1 - L
                B = T
                m = n
        R = (R << 1) | int(sn)
    return L


PI_LINEAR = [1 / 96, 1 / 32, 1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 48]   # точные значения 0.010417, 0.03125, ...


# константы референсного кода STS 2.1.x (там π_0 = 0.01047 - опечатка; нужны только для сверки с Таблицей 1)
PI_LINEAR_STS = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]


def linear_complexity_test(bits, M=500, K=6, pi=None):
    """
    NIST 2.10.  N = n // M блоков; L_i - линейная сложность i-го блока;
    mu = M/2 + (9 + (-1)^(M+1))/36 - (M/3 + 2/9)/2^M;
    T_i = (-1)^M (L_i - mu) + 2/9 -> классы v_0..v_6 по порогам -2.5,-1.5,-0.5,0.5,1.5,2.5;
    chi2 = sum (v_i - N pi_i)^2/(N pi_i); P = igamc(K/2, chi2/2).
    """
    pi = PI_LINEAR if pi is None else pi
    b = np.asarray(bits, dtype=np.uint8)
    N = len(b) // M
    mu = M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / 2.0 ** M
    v = [0] * (K + 1)
    for i in range(N):
        L = berlekamp_massey(b[i * M:(i + 1) * M].tolist())
        T = (-1) ** M * (L - mu) + 2.0 / 9.0
        if T <= -2.5:   v[0] += 1
        elif T <= -1.5: v[1] += 1
        elif T <= -0.5: v[2] += 1
        elif T <= 0.5:  v[3] += 1
        elif T <= 1.5:  v[4] += 1
        elif T <= 2.5:  v[5] += 1
        else:           v[6] += 1
    chi2 = sum((v[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K + 1))
    return float(gammaincc(K / 2.0, chi2 / 2.0)), {"v": v, "chi2": chi2, "N": N, "mu": mu}


# --------------------------------------------------------------------------- утилиты
def read_bits(path, n=None):
    """Читает файл с цифрами 0/1 (пробелы и переводы строк игнорируются)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    arr = arr[(arr == 48) | (arr == 49)] - 48
    return arr[:n] if n else arr
