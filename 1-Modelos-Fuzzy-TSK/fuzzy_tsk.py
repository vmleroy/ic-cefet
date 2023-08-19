import sympy as sp
import sympy.plotting as spplt

import numpy as np
import matplotlib.pyplot as plt


def weighted_average(w1: float, y1: float, w2: float, y2: float) -> float:
    return (w1 * y1 + w2 * y2) / (w1 + w2)


def error(y: float, yd: float) -> float:
    return (1 / 2) * (y - yd) ** 2


def example_4():
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


def example_3():
    generation = 1000
    learning_rate = 0.01

    x_range = np.linspace(-2, 2, 100)

    points = np.linspace(-2, 2, 100)
    graphs_result = []
    
    standard_deviation_1 = 1
    standard_deviation_2 = 1

    mean_1 = -2
    mean_2 = 2

    p1 = np.random.sample()
    q1 = np.random.sample()

    p2 = np.random.sample()
    q2 = np.random.sample()

    for i in range(generation):
        np.random.shuffle(x_range)
        for x in x_range:
            yd = x ** 2

            y1 = p1*x + q1
            y2 = p2*x + q2

            w1 = np.exp( (-1/2) * ( (x - mean_1) / standard_deviation_1 )**2 )
            w2 = np.exp( (-1/2) * ( (x - mean_2) / standard_deviation_2 )**2 )

            y = weighted_average(w1, y1, w2, y2)

            mean_1 = mean_1 - learning_rate * (y - yd) * w2 * ( (y1 - y2) / (w1 + w2)**2 ) * w1 * ( (x - mean_1) / (standard_deviation_1**2) )
            mean_2 = mean_2 - learning_rate * (y - yd) * w1 * ( (y2 - y1) / (w1 + w2)**2 ) * w2 * ( (x - mean_2) / (standard_deviation_2**2) )

            standard_deviation_1 = standard_deviation_1 - learning_rate * (y - yd) * w2 * ( (y1 - y2) / (w1 + w2)**2 ) * w1 * ( (x - mean_1)**2 / (standard_deviation_1**3) )
            standard_deviation_2 = standard_deviation_2 - learning_rate * (y - yd) * w1 * ( (y2 - y1) / (w1 + w2)**2 ) * w2 * ( (x - mean_2)**2 / (standard_deviation_2**3) )
            
            p1 = p1 - learning_rate * (y - yd) * ( w1 / (w1 + w2) ) * x
            p2 = p2 - learning_rate * (y - yd) * ( w2 / (w1 + w2) ) * x

            q1 = q1 - learning_rate * (y - yd) * ( w1 / (w1 + w2) )
            q2 = q2 - learning_rate * (y - yd) * ( w2 / (w1 + w2) )

        y1 = p1*points + q1
        y2 = p2*points + q2

        w1 = np.exp( (-1/2) * ( (points - mean_1) / standard_deviation_1 )**2 )
        w2 = np.exp( (-1/2) * ( (points - mean_2) / standard_deviation_2 )**2 )

        result_prediction = weighted_average(w1, y1, w2, y2)
        e = error(y, x**2)

        graphs_result.append([result_prediction, e])
        # print(f"Generation: {i} --> Error: {e}")

    # plt.plot(x_range, graphs_result[-1], label="Curva da função y(x)=x2")
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    for i in np.logspace(0, np.log10(len(graphs_result)), num=9, endpoint=True).astype(int)-1:
        ax.plot(points, graphs_result[i][0], label=f'Gen {i+1}',)
    ax.plot(points, points**2, label='True f(x)', linestyle='--', linewidth=2)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.8), ncol=2, fancybox=True, shadow=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fuzzy TSK model evolution over 1000 generations')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    ax.plot(points, graphs_result[-1][0], label="Fuzzy tsk model")
    ax.plot(points, points**2, label="True f(x)", linestyle='--', linewidth=2)
    ax.annotate(f"Error: {graphs_result[-1][1]}", xy=(0.35, 0.5), xycoords='axes fraction', fontsize=10)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.8), ncol=2, fancybox=True, shadow=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fuzzy TSK model vs True f(x)')
    plt.show()

        

    
        

            

