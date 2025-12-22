import matplotlib.pyplot as plt

def create_placeholder_logo(filename, text, color='blue'):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, text, fontsize=20, ha='center', va='center', color=color, weight='bold')
    ax.axis('off')
    # Save as transparent PNG
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Created {filename}")

if __name__ == '__main__':
    create_placeholder_logo('images/columbia_logo.png', 'Columbia\nUniversity', '#1D4F91')
    create_placeholder_logo('images/columbia_engineering_logo.png', 'Columbia\nEngineering', '#1D4F91')
