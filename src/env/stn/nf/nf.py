from src.env.stn.nf.__dep import *


class NoisyFunction:
    def __init__(self, nugget, slope, sill, sigma):
        self.nugget = nugget
        self.slope = slope
        self.sill = sill
        self.sigma = sigma

    def true_value(self, x):
        if x <= self.nugget:
            return 0
        elif x < self.nugget + self.sill/self.slope:
            return self.slope * (x - self.nugget)
        else:
            return self.sill

    def sample_at(self, x):
        if x <= self.nugget:
            noise = 0
        elif x < self.nugget + self.sill/self.slope:
            noise = normal(loc=0, scale=self.sigma * (x - self.nugget)/(self.sill/self.slope))
        else:
            noise = normal(loc=0, scale=self.sigma)
        return self.true_value(x) + noise

    def __call__(self, x, true_value=False):
        return self.true_value(x) if true_value else self.sample_at(x)

    def show(self, x_range, res=1000, samples=0):
        x_res = np.linspace(start=x_range[0], stop=x_range[1], num=res)
        y_res = [self.true_value(x) for x in x_res]
        x_smp = choice(x_res, samples, replace=False)
        y_smp = [self.sample_at(x) for x in x_smp]
        plt.figure(0)
        plt.grid()
        plt.plot(x_res, y_res)
        plt.scatter(x_smp, y_smp, s=5)
        plt.show()
