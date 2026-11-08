# stock/simulation.py
import pandas as pd

def monte_carlo(stock, sigma, r, delta_t, n_steps, n_simulations):
    results = []
    for _ in range(n_simulations):
        path = stock.generate_path(sigma, r, delta_t, n_steps)
        results.append(path)
    
    df = pd.DataFrame(results).T
    df.index.name = "Time Step"
    
    return df   # <-- MUST RETURN!