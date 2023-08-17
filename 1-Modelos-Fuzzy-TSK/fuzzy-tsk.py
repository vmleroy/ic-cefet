import sympy as sp
import sympy.plotting as spplt


def weighted_average(w1: float, y1: float, w2: float, y2: float) -> float:
    return (w1 * y1 + w2 * y2) / (w1 + w2)


def error(y: float, yd: float) -> float:
    return (1 / 2) * (y - yd) ** 2


def two_lines():
    # Definindo os simbolos
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    # Definindo variaveis das retas
    a1 = -1 / 4
    a2 = 1 / 4
    b1 = 1 / 2
    b2 = 1 / 2

    # Definindo variaveis de entrada
    w1 = a1 * x + b1
    w2 = a2 * x + b2
    print(f"w1 = {w1}")
    print(f"w2 = {w2}")

    y1 = -2 * x
    y2 = 2 * x
    print()
    print(f"y1 = {y1}")
    print(f"y2 = {y2}")

    y = weighted_average(w1, y1, w2, y2)
    result = sp.simplify(y)
    print()
    print(f"y = {y}")
    print(f"y = {result}")

    p1 = spplt.plot(x**2, title="Curva da função y(x)=x2 e Curva estimada ajustada", xlabel="x", ylabel="y", show=False)
    p2 = spplt.plot(y, xlabel="x", ylabel="y", show=False)
    p1.append(p2[0])
    p1.show()


def gaussian():
    pass


def main():
    two_lines()


if __name__ == "__main__":
    main()
