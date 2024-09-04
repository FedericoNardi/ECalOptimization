import numpy as np
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

class Shower:
    def __init__(self, energy, *args, **kwargs):
        self.energy = torch.tensor(energy, dtype=torch.float32, requires_grad=False)
        # Define a meshgrid x in (-25,25) z in (0,500)
        self.x = torch.linspace(-25, 25, 100, requires_grad=False)
        self.z = torch.linspace(0, 500, 100, requires_grad=False) + 0.01
        self.X, self.Z = torch.meshgrid(self.x, self.z, indexing='ij')
        
        # Load parameters and convert to tensors
        self._pars_a = torch.tensor(np.loadtxt('modules/pars/fit_pars_final_0.txt'), dtype=torch.float32, requires_grad=False)
        self._pars_b = torch.tensor(np.loadtxt('modules/pars/fit_pars_final_1.txt'), dtype=torch.float32, requires_grad=False)
        self._pars_c = torch.tensor(np.loadtxt('modules/pars/fit_pars_final_2.txt'), dtype=torch.float32, requires_grad=False)
        
        self._a, self._b, self._c = self.calculate_coefficients(self.Z, **kwargs)

    def calculate_coefficients(self, z, *args, DEBUG=False):
        def func0(x, a, b, c):
            return a * (b - torch.exp(-c * x))
        
        def func1(x, a, b, c):
            return a * torch.exp(-b * x) + c
        
        def func2(x, a, b, c):
            return a * x**2 + b * x + c
            
        a0 = func0(self.energy, *self._pars_a[0])
        a1 = func1(self.energy, *self._pars_a[1])
        a2 = func0(self.energy, *self._pars_a[2])
        a3 = func0(self.energy, *self._pars_a[3])

        b0 = func1(self.energy, *self._pars_b[0])
        b1 = func1(self.energy, *self._pars_b[1])
        b2 = func1(self.energy, *self._pars_b[2])

        c0 = func1(self.energy, *self._pars_c[0])
        c1 = func1(self.energy, *self._pars_c[1])
        c2 = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)  # Use torch constant for c2

        # Plot c0, c1, c2 for a range of energies in 0.5, 150
        if DEBUG:
            print('Producing debug plots for c parameters...')
            xx = torch.linspace(0.5, 150, 100, requires_grad=False)
            par0 = torch.stack([func1(x, *self._pars_c[0]) for x in xx])
            par1 = torch.stack([func1(x, *self._pars_c[1]) for x in xx])
            par2 = torch.stack([func2(x, *self._pars_c[2]) for x in xx])
            
            plt.figure(figsize=(10, 15))
            plt.subplot(3, 1, 1)
            plt.plot(xx.numpy(), par0.numpy())
            plt.title('c0(E)')
            plt.subplot(3, 1, 2)
            plt.plot(xx.numpy(), par1.numpy())
            plt.title('c1(E)')
            plt.subplot(3, 1, 3)
            plt.plot(xx.numpy(), par2.numpy())
            plt.title('c2(E)')
            plt.savefig('img/debug/c_parameters.png')
            plt.close()

        # same for b0, b1, b2
        if DEBUG:
            print('Producing debug plots for b parameters...')
            xx = torch.linspace(0.5, 150, 100, requires_grad=False)
            par0 = torch.stack([func1(x, *self._pars_b[0]) for x in xx])
            par1 = torch.stack([func1(x, *self._pars_b[1]) for x in xx])
            par2 = torch.stack([func1(x, *self._pars_b[2]) for x in xx])
            
            plt.figure(figsize=(10, 15))
            plt.subplot(3, 1, 1)
            plt.plot(xx.numpy(), par0.numpy())
            plt.title('b0(E)')
            plt.subplot(3, 1, 2)
            plt.plot(xx.numpy(), par1.numpy())
            plt.title('b1(E)')
            plt.subplot(3, 1, 3)
            plt.plot(xx.numpy(), par2.numpy())
            plt.title('b2(E)')
            plt.savefig('img/debug/b_parameters.png')
            plt.close()

        # same for a0, a1, a2, a3
        if DEBUG:
            print('Producing debug plots for a parameters...')
            xx = torch.linspace(0.5, 150, 100, requires_grad=False)
            par0 = torch.stack([func0(x, *self._pars_a[0]) for x in xx])
            par1 = torch.stack([func1(x, *self._pars_a[1]) for x in xx])
            par2 = torch.stack([func0(x, *self._pars_a[2]) for x in xx])
            par3 = torch.stack([func0(x, *self._pars_a[3]) for x in xx])
            
            plt.figure(figsize=(10, 15))
            plt.subplot(4, 1, 1)
            plt.plot(xx.numpy(), par0.numpy())
            plt.title('a0(E)')
            plt.subplot(4, 1, 2)
            plt.plot(xx.numpy(), par1.numpy())
            plt.title('a1(E)')
            plt.subplot(4, 1, 3)
            plt.plot(xx.numpy(), par2.numpy())
            plt.title('a2(E)')
            plt.subplot(4, 1, 4)
            plt.plot(xx.numpy(), par3.numpy())
            plt.title('a3(E)')
            plt.savefig('img/debug/a_parameters.png')
            plt.close()

        _a = a0 * torch.exp(a1 * z**3 + a2 * z**2 + a3 * z)
        _b = b0 * torch.exp(-b1 * z) + b2
        _c = c0 * torch.exp(-c1 * (torch.log(z + 0.01) - c2)**2)

        return _a, _b, _c

    def __call__(self):
        return torch.exp(self._a * torch.exp(-self._b * torch.abs(self.X))) + self._c - 1.0


    def plot_parameters(self):
        z = np.linspace(0,500,100)
        plt.figure(figsize=(10,15))
        a, b, c = self.calculate_coefficients(z)
        plt.subplot(3,1,1)
        plt.plot(z, a)
        plt.title('a(z)')
        plt.subplot(3,1,2)
        plt.plot(z, b)
        plt.title('b(z)')
        plt.subplot(3,1,3)
        plt.plot(z, c)
        plt.title('c(z)')
        plt.savefig('img/parameters.png')
        plt.close()