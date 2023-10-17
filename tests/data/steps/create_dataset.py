import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


class System:
    def __init__(self, k: float, tau: float):
        self.a: float = -1/tau
        self.b: float = k/tau

    def __call__(self, u: np.ndarray, t_step: float) -> np.ndarray:
        y: np.ndarray = np.zeros(len(u))
        for i in range(len(u)-1):
            y[i+1] = (1 + t_step*self.a)*y[i] + t_step*self.b*u[i]

        return y


def main() -> None:
    k: float = 1
    tau: float = 0.5
    system: System = System(k, tau)

    t_step: float = 0.1
    t_end: float = 10
    n_signals: int = 100
    for i in range(n_signals):
        u: np.ndarray = np.ones(int(t_end/t_step))

        u[:len(u)//2] = np.random.rand()
        u[len(u)//2:] = np.random.rand()

        y: np.ndarray = system(u, t_step)

        # plt.plot(u)
        # plt.plot(y)
        # plt.show()

        pd.DataFrame({'input': u, 'output': y}).to_csv(f'data/steps/step_{i}.csv', index=False)


if __name__ == '__main__':
    main()
