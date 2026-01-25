import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression
import warnings

class AmericanBarrierOption:
    def __init__(self, S0: float, K: float, T: float, r: float, 
                 sigma: float, barrier: float, barrier_type: str = 'up-and-out',
                 option_type: str = 'call', dividend: float = 0.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.dividend = dividend
        
        if barrier_type == 'up-and-out' and S0 >= barrier:
            warnings.warn("Initial price above up-and-out barrier: option is worthless")
        if barrier_type == 'down-and-out' and S0 <= barrier:
            warnings.warn("Initial price below down-and-out barrier: option is worthless")
    
    def simulate_paths(self, num_paths: int, num_steps: int, 
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        
        dt = self.T / num_steps
        time_grid = np.linspace(0, self.T, num_steps + 1)
        
        S = np.zeros((num_paths, num_steps + 1))
        S[:, 0] = self.S0
        
        Z = np.random.normal(0, 1, (num_paths, num_steps))
        drift = (self.r - self.dividend - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        for i in range(num_steps):
            S[:, i + 1] = S[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        if self.barrier_type in ['up-and-out', 'up-and-in']:
            M = np.maximum.accumulate(S, axis=1)
        else:
            M = np.minimum.accumulate(S, axis=1)
        
        return S, M
    
    def intrinsic_value(self, S: np.ndarray, M: np.ndarray, 
                       time_idx: int) -> np.ndarray:
        if self.option_type == 'call':
            payoff = np.maximum(S - self.K, 0)
        else:
            payoff = np.maximum(self.K - S, 0)
        
        barrier_hit = np.zeros_like(payoff, dtype=bool)
        if self.barrier_type == 'up-and-out':
            barrier_hit = M >= self.barrier
        elif self.barrier_type == 'down-and-out':
            barrier_hit = M <= self.barrier
        elif self.barrier_type == 'up-and-in':
            barrier_hit = M < self.barrier
        else:
            barrier_hit = M > self.barrier
        
        payoff[barrier_hit] = 0
        return payoff
    
    def laguerre_polynomials(self, x: np.ndarray, degree: int) -> np.ndarray:
        n = len(x)
        basis = np.zeros((n, degree + 1))
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = 1.0 - x
        if degree >= 2:
            basis[:, 2] = 0.5 * (x ** 2 - 4 * x + 2)
        if degree >= 3:
            for j in range(3, degree + 1):
                basis[:, j] = ((2 * j - 1 - x) * basis[:, j - 1] - 
                              (j - 1) * basis[:, j - 2]) / j
        return basis
    
    def price_lsm(self, num_paths: int = 10000, num_steps: int = 252,
                  basis_degree: int = 3, seed: Optional[int] = None) -> Tuple[float, float]:
        S, M = self.simulate_paths(num_paths, num_steps, seed)
        
        dt = self.T / num_steps
        time_grid = np.linspace(0, self.T, num_steps + 1)
        
        cash_flows = np.zeros((num_paths, num_steps + 1))
        exercise_times = np.full(num_paths, num_steps)
        
        intrinsic = self.intrinsic_value(S, M, num_steps)
        cash_flows[:, num_steps] = intrinsic[:, num_steps]
        
        for i in range(num_steps - 1, -1, -1):
            intrinsic_i = self.intrinsic_value(S, M, i)
            in_the_money = intrinsic_i[:, i] > 0
            
            if self.barrier_type == 'up-and-out':
                alive = M[:, i] < self.barrier
            elif self.barrier_type == 'down-and-out':
                alive = M[:, i] > self.barrier
            elif self.barrier_type == 'up-and-in':
                alive = M[:, i] >= self.barrier
            else:
                alive = M[:, i] <= self.barrier
            
            exercise_candidates = in_the_money & alive
            
            if np.sum(exercise_candidates) == 0:
                continue
            
            continuation_values = np.zeros(num_paths)
            for m in range(num_paths):
                if exercise_candidates[m]:
                    future_cf = cash_flows[m, i + 1:]
                    if np.any(future_cf > 0):
                        exercise_idx = np.argmax(future_cf > 0) + i + 1
                        discount_factor = np.exp(-self.r * (time_grid[exercise_idx] - time_grid[i]))
                        continuation_values[m] = discount_factor * cash_flows[m, exercise_idx]
            
            X = S[exercise_candidates, i]
            y = continuation_values[exercise_candidates]
            
            if len(X) > basis_degree + 1:
                basis = self.laguerre_polynomials(X / self.K, basis_degree)
                model = LinearRegression()
                model.fit(basis, y)
                estimated_continuation = model.predict(basis)
            else:
                estimated_continuation = y
            
            exercise_now = intrinsic_i[exercise_candidates, i] > estimated_continuation
            
            exercise_indices = np.where(exercise_candidates)[0]
            for idx, should_exercise in zip(exercise_indices, exercise_now):
                if should_exercise:
                    cash_flows[idx, i] = intrinsic_i[idx, i]
                    cash_flows[idx, i + 1:] = 0
                    exercise_times[idx] = i
        
        discounted_payoffs = np.zeros(num_paths)
        for m in range(num_paths):
            if exercise_times[m] < num_steps + 1:
                discount = np.exp(-self.r * time_grid[exercise_times[m]])
                discounted_payoffs[m] = discount * cash_flows[m, exercise_times[m]]
        
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
        
        return option_price, std_error
    
    def price_european_barrier(self, num_paths: int = 100000,
                              seed: Optional[int] = None) -> Tuple[float, float]:
        S, M = self.simulate_paths(num_paths, 252, seed)
        intrinsic = self.intrinsic_value(S, M, -1)
        payoffs = intrinsic[:, -1]
        discounted = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(num_paths)
        return price, std_error
