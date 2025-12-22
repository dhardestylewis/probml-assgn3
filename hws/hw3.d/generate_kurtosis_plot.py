
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

def generate_kurtosis_plot():
    # Target excess kurtosis approx 9.27
    # For Student-t, excess K = 6/(nu-4)  => 9.27 = 6/(nu-4) => nu-4 = 0.647 => nu = 4.647
    nu = 4.65
    
    x = np.linspace(-6, 6, 1000)
    
    # Normal distribution (Kurtosis = 0)
    y_norm = norm.pdf(x, 0, 1)
    
    # Student-t distribution (Kurtosis approx 9.3)
    # We standardize it to have unit variance for fair comparison like residuals
    # Variance of t is nu/(nu-2). Scale factor = sqrt((nu-2)/nu)
    scale = np.sqrt((nu-2)/nu)
    y_t = t.pdf(x, df=nu, scale=scale)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_norm, 'k--', label='Normal (Kurt=0)', linewidth=2, alpha=0.7)
    plt.plot(x, y_t, 'b-', label=f'Heavy Tails (Kurt approx 9.3)', linewidth=2)
    
    plt.fill_between(x, y_t, y_norm, where= (np.abs(x)>2), color='blue', alpha=0.1, label='Excess Mass in Tails')
    
    plt.title('Visualizing Kurtosis: Normal vs Heavy-Tailed')
    plt.xlabel('Standard Deviations')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = 'hws/hw3.d/images/kurtosis_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_kurtosis_plot()
