def add_graded_density_contours(ax, x, y, xlim, ylim, levels=5):
    """
    Adds graded density contours to an existing plot.
    Innermost lines are thicker and more opaque than outermost lines.
    """
    try:
        import scipy.stats as st
        # Create a 2D histogram for contouring (faster robust method)
        # Or use KDE if N is small? Let's use KDE for smoothness if N < 10000
        # But here N ~ 30k+. KDE might be slow. 
        # Using 2D histogram approach embedded in a grid is robust.
        
        # Grid setup
        deltaX = (xlim[1] - xlim[0]) / 50
        deltaY = (ylim[1] - ylim[0]) / 50
        xmin, xmax = xlim
        ymin, ymax = ylim
        X_grid, Y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        values = np.vstack([x, y])
        
        # Use gaussian_kde
        # For speed with large N, we can downsample or use binned approximation
        if len(x) > 10000:
             # Downsample for KDE estimation speed
             idx = np.random.choice(len(x), 5000, replace=False)
             values_kde = values[:, idx]
        else:
             values_kde = values
             
        kernel = st.gaussian_kde(values_kde)
        Z_grid = np.reshape(kernel(positions).T, X_grid.shape)
        
        # Define styles for levels (Outer -> Inner)
        # We want inner to be thicker/darker
        linewidths = np.linspace(1.0, 3.0, levels)
        alphas = np.linspace(0.4, 1.0, levels)
        colors = [(0, 0, 0, a) for a in alphas]
        
        ax.contour(X_grid, Y_grid, Z_grid, levels=levels, 
                   linewidths=linewidths, colors=colors, zorder=3)
        
        # Annotation/Legend
        # Add a text box explaining the lines
        ax.text(0.02, 0.98, "Contours: Density Iso-lines\n(Thicker = Higher Density)",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3),
                zorder=4)
                
    except Exception as e:
        print(f"[Eval] Failed to add density contours: {e}")
