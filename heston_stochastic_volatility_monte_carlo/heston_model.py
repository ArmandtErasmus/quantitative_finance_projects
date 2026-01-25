import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.stats import ncx2
from scipy.optimize import minimize
import warnings

class HestonModel:
    def __init__(self, kappa: float, theta: float, sigma_v: float,
                 rho: float, v0: float, r: float = 0.05):
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        self.r = r
        
        if 2 * kappa * theta < sigma_v ** 2:
            warnings.warn("Feller condition may be violated")
    
    def exact_variance_step(self, v_t: float, dt: float,
                           random_vars: np.ndarray) -> np.ndarray:
        c = self.sigma_v ** 2 * (1 - np.exp(-self.kappa * dt)) / (4 * self.kappa)
        d = 4 * self.kappa * self.theta / (self.sigma_v ** 2)
        noncentrality = 4 * self.kappa * np.exp(-self.kappa * dt) / \
                       (self.sigma_v ** 2 * (1 - np.exp(-self.kappa * dt))) * v_t
        
        v_next = c * ncx2.rvs(d, noncentrality, size=len(random_vars))
        return v_next
    
    def simulate_path_exact(self, S0: float, T: float, n_steps: int,
                           num_paths: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        S = np.zeros((num_paths, n_steps + 1))
        v = np.zeros((num_paths, n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = self.v0
        
        for i in range(n_steps):
            Z_v = np.random.normal(0, 1, num_paths)
            Z_S = np.random.normal(0, 1, num_paths)
            Z_corr = self.rho * Z_v + np.sqrt(1 - self.rho ** 2) * Z_S
            
            v_t = v[:, i]
            v_next = self.exact_variance_step(v_t, dt, Z_v)
            
            v_bar = (v_t + v_next) / 2
            sqrt_v_bar = np.sqrt(np.maximum(v_bar, 0))
            
            K0 = -self.rho * self.kappa * self.theta * dt / self.sigma_v
            K1 = (self.rho * self.kappa / self.sigma_v - 0.5) * dt - self.rho / self.sigma_v
            K2 = self.rho / self.sigma_v
            K3 = (1 - self.rho ** 2) * dt
            
            log_S_next = (np.log(S[:, i]) + self.r * dt + 
                         K0 + K1 * v_t + K2 * v_next +
                         sqrt_v_bar * np.sqrt(K3) * Z_S)
            
            S[:, i + 1] = np.exp(log_S_next)
            v[:, i + 1] = v_next
        
        return S, v
    
    def characteristic_function(self, u: float, T: float, S0: float) -> complex:
        i = 1j
        kappa = self.kappa
        theta = self.theta
        sigma_v = self.sigma_v
        rho = self.rho
        v0 = self.v0
        
        d = np.sqrt((rho * sigma_v * i * u - kappa) ** 2 + sigma_v ** 2 * (i * u + u ** 2))
        g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)
        
        C = self.r * i * u * T + kappa * theta / (sigma_v ** 2) * \
            ((kappa - rho * sigma_v * i * u - d) * T - 
             2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        
        D = (kappa - rho * sigma_v * i * u - d) / (sigma_v ** 2) * \
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        
        return np.exp(C + D * v0 + i * u * np.log(S0))
    
    def price_european_call_fft(self, S0: float, K: float, T: float,
                                N: int = 4096, alpha: float = 1.5) -> float:
        i = 1j
        eta = 0.25
        b = N * eta / 2
        u = np.arange(1, N + 1) * eta
        
        kappa = np.log(K)
        v = u - (alpha + 1) * i
        phi = self.characteristic_function(v, T, S0)
        psi = np.exp(-self.r * T) * phi / ((alpha + i * u) * (alpha + 1 + i * u))
        
        integral = np.exp(-alpha * kappa) / np.pi * np.sum(
            np.real(psi * np.exp(-i * u * kappa)) * eta
        )
        
        return S0 - K * np.exp(-self.r * T) * integral
    
    def price_european_call_mc(self, S0: float, K: float, T: float,
                               num_paths: int = 10000, n_steps: int = 252) -> Tuple[float, float]:
        S, _ = self.simulate_path_exact(S0, T, n_steps, num_paths)
        
        payoffs = np.maximum(S[:, -1] - K, 0)
        price = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error

class HestonOptionPricer:
    def __init__(self, model: HestonModel):
        self.model = model
    
    def price_asian_call(self, S0: float, K: float, T: float,
                        num_paths: int = 10000, n_steps: int = 252) -> Tuple[float, float]:
        S, _ = self.model.simulate_path_exact(S0, T, n_steps, num_paths)
        
        asian_price = np.mean(S, axis=1)
        payoffs = np.maximum(asian_price - K, 0)
        price = np.exp(-self.model.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.model.r * T) * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error
    
    def price_barrier_up_and_out_call(self, S0: float, K: float, B: float, T: float,
                                     num_paths: int = 10000, n_steps: int = 252) -> Tuple[float, float]:
        S, _ = self.model.simulate_path_exact(S0, T, n_steps, num_paths)
        
        max_price = np.max(S, axis=1)
        knock_out = max_price >= B
        payoffs = np.maximum(S[:, -1] - K, 0)
        payoffs[knock_out] = 0
        
        price = np.exp(-self.model.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.model.r * T) * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error
