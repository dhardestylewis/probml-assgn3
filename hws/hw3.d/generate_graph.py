import graphviz

def create_miwae_graph():
    dot = graphviz.Digraph('miwae', comment='Semi-Supervised MIWAE')
    dot.attr(rankdir='LR')
    dot.attr(splines='false')
    
    # Global Parameters (theta)
    dot.node('theta', r'&theta;', shape='circle', style='filled', fillcolor='white')
    
    # Plate for N data points
    with dot.subgraph(name='cluster_N') as c:
        c.attr(label='N', labeljust='r', labelloc='b')
        c.attr(style='rounded')
        
        # Latent Variable z
        c.node('z', 'z', shape='circle', style='filled', fillcolor='white')
        
        # Mixture Component c (marginalized/implicit in some views, but explicit in generative)
        # We can represent the mixture implicitly in z or explicitly. 
        # Given the "Mixture of Student-t" description, z depends on params.
        # Let's keep it simple as Z -> X, Z -> Y for the core MIWAE structure.
        
        # Observed Covariates x (Shaded)
        c.node('x', 'x', shape='circle', style='filled', fillcolor='lightgrey')
        
        # Observed Price y (Shaded/Partially) - We'll shadow it to indicate semi-supervised
        # or just shaded and note missingness in text. 
        # Standard semi-supervised notation: y is sometimes unobserved. 
        # But for the graph, we usually show the structure.
        c.node('y', 'y', shape='circle', style='filled', fillcolor='lightgrey')
        
        # Edges
        # z -> x (decoder)
        c.edge('z', 'x')
        # z -> y (predictor/head)
        c.edge('z', 'y')

    # Parameters to Latents/Data
    dot.edge('theta', 'z')
    dot.edge('theta', 'x')
    dot.edge('theta', 'y')
    
    # Render
    dot.render('images/miwae_graphical_model_clean', format='png', cleanup=True)
    print("Graph generated: images/miwae_graphical_model_clean.png")

if __name__ == '__main__':
    create_miwae_graph()
