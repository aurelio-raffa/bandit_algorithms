from source.random_function.__dependencies import *


class RandomFunction:
    def __init__(self, range_x, scale_y, sigma, seed=None):
        if seed is not None:
            np.random.seed(seed)
        parameters = np.apply_along_axis(func1d=lambda x: 2*(x-.5), axis=0, arr=sample(15))
        self.__parameters = parameters
        self.range_x = range_x
        self.range_y = scale_y
        self.sigma = sigma

        def internal_function(x):
            scaled_x = (x - range_x[0]) / (range_x[1] - range_x[0])
            ans = 0
            for i in range(6):
                ans += parameters[i] * (scaled_x ** i)
            for i in range(2):
                ans += .5 * parameters[6 + 3*i] * np.sin(40 * parameters[7 + 3*i] * scaled_x + parameters[8 + 3*i])
            exp = np.exp(5 * (parameters[13] + 1) * scaled_x + parameters[14])
            ans += 7 * (parameters[12] + 1) * exp/(1 + exp)
            return ans / 8

        self.__internal_function = internal_function

    def true_value(self, x):
        return (self.range_y[1] - self.range_y[0]) * self.__internal_function(x) + self.range_y[0]

    def sample_at(self, x):
        return self.true_value(x) + np.random.normal(scale=self.sigma)

    def __call__(self, x, true_value=False):
        return self.true_value(x) if true_value else self.sample_at(x)

    def show(self, res=1000, samples=0):
        x_res = np.linspace(start=self.range_x[0], stop=self.range_x[1], num=res)
        x_smp = [(self.range_x[1] - self.range_x[0]) * x + self.range_x[0] for x in sample(samples)]
        y_res = [self.true_value(x) for x in x_res]
        y_smp = [self.sample_at(x) for x in x_smp]
        sns.scatterplot(x=x_smp, y=y_smp)
        sns.lineplot(x=x_res, y=y_res)
        plt.show()
