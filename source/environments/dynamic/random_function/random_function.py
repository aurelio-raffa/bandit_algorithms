from source.environments.dynamic.random_function.__dependencies import *


class RandomFunction:
    def __init__(self, range_x, scale_y, sigma, seed=None):
        if seed is not None:
            np.random.seed(seed)
        parameters = sample(15)
        self.__parameters = parameters
        self.range = range_x
        self.scale = scale_y
        self.sigma = sigma

        def internal_function(x):
            scaled_x = (x - range_x[0]) / (range_x[1] - range_x[0])
            ans = 1
            for i in range(6):
                ans *= (parameters[i] - scaled_x)
            ans = .15 * ans + .85
            for i in range(2):
                ans *= \
                    (.95 + .05 * parameters[6 + 3*i] * np.sin(
                        20 * parameters[7 + 3*i] * scaled_x + parameters[8 + 3*i]))
            exp = np.exp(20 * (parameters[12] * (scaled_x - parameters[13]/2)))
            ans = .75 * parameters[14] * exp/(1 + exp) + .25 * ans
            return ans

        self.__internal_function = internal_function

    def true_value(self, x):
        return (self.scale[1] - self.scale[0]) * self.__internal_function(x) + self.scale[0]

    def sample_at(self, x):
        return self.true_value(x) + np.random.normal(scale=self.sigma)

    def __call__(self, x, true_value=False):
        return self.true_value(x) if true_value else self.sample_at(x)

    def show(self, res=1000, samples=0):
        x_res = np.linspace(start=self.range[0], stop=self.range[1], num=res)
        x_smp = [(self.range[1] - self.range[0]) * x + self.range[0] for x in sample(samples)]
        y_res = [self.true_value(x) for x in x_res]
        y_smp = [self.sample_at(x) for x in x_smp]
        plt.figure(0)
        plt.grid()
        plt.axes(ylim=(0, 1))
        plt.plot(x_res, y_res)
        plt.plot(x_smp, y_smp)
        plt.show()
