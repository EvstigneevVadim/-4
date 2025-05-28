import numpy as np
import matplotlib.pyplot as plt
def load_matrix():
    A = np.loadtxt("arr.txt", dtype=int)
    if A.shape[0] != A.shape[1] or A.shape[0] % 2 != 0:
        print("Ошибка: файл должен содержать квадратную матрицу чётного размера")
        exit()
    return A
def build_F(A, K):
    F = A.copy()
    n = A.shape[0]
    h = n // 2
    E = A[:h, :h]
    B = A[:h, h:]
    C = A[h:, h:]
    zero_count = np.sum(E[:, ::2] == 0)
    rows = E[::2, :]
    non_zero = rows[rows != 0]
    product = np.prod(non_zero) if non_zero.size > 0 else 0
    kz = zero_count * K
    print(f"\nКоличество нулей в нечётных столбцах E: {zero_count}")
    print(f"K * количество нулей: {K} * {zero_count} = {kz}")
    print(f"Произведение чисел в нечётных строках E: {product}")
    if kz > product:
        print("Меняем B и C симметрично")
        F[:h, h:], F[h:, h:] = C.copy(), B.copy()
    else:
        temp = F[:h, :h].copy()
        F[:h, :h] = np.flip(F[:h, h:], axis=(0, 1))
        F[:h, h:] = np.flip(temp, axis=(0, 1))
        print("Меняем B и E несимметрично")
    return F
def compute_result(A, F, K):
    det_A = np.linalg.det(A)
    diag_sum_F = np.trace(F) + np.trace(np.fliplr(F))
    print(f"\nОпределитель матрицы A: {det_A:.2f}")
    print(f"Сумма диагоналей матрицы F: {diag_sum_F}")
    if np.isclose(det_A, 0) or np.isclose(np.linalg.det(F), 0):
        return None, None, "Ошибка: одна из матриц необратима"
    if det_A > diag_sum_F:
        res = A @ A.T - K * np.linalg.inv(F)
        err1 = "A @ A.T - K * F^-1"
        return res, err1, None
    else:
        G = np.tril(A)
        res = (np.linalg.inv(A) + G - F.T) * K
        err1 = "(A^-1 + G - F.T) * K"
        return res, err1, None
def plot_graphs(F):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(F.sum(axis=1), 'o-')
    axs[0].set_title("Суммы по строкам")
    axs[1].bar(range(F.shape[1]), np.median(F, axis=0))
    axs[1].set_title("Медиана по столбцам")
    axs[2].hist(F.flatten(), bins=10, density=True)
    axs[2].set_title("Плотность значений")
    for ax in axs:
        ax.grid()
    plt.tight_layout()
    plt.show()
def main():
    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
    K = int(input("Введите K: "))
    A = load_matrix()
    print("\nМатрица A:\n", A)
    F = build_F(A, K)
    print("\nМатрица F:\n", F)
    res, err1, err = compute_result(A, F, K)
    if err:
        print(err)
    else:
        print(f"\nРезультат выражения ({err1}):\n", res)
    plot_graphs(F)
main()