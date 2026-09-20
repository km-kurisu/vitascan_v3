import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. Execute pipeline and capture objects globally
gnn_model, test_graph, y_test, final_preds, target_names_with_grey = run_full_hybrid_pipeline(
    alpha=0.50,
    lower_bound=0.35,
    upper_bound=0.65
)

# 2. t-SNE Visualizer Function
def visualize_gnn_embeddings_tsne(gnn_model, test_graph, y_test_labels):
    gnn_model.eval()
   
    with torch.no_grad():
        x_dict = test_graph.x_dict
        edge_index_dict = test_graph.edge_index_dict

        # Forward projection through feature layers
        x_dict['patient'] = F.relu(gnn_model.patient_proj(x_dict['patient']))
        x_dict['biomarker'] = F.relu(gnn_model.biomarker_proj(x_dict['biomarker']))
        x_dict['etiology'] = F.relu(gnn_model.etiology_proj(x_dict['etiology']))

        # Heterogeneous Message Passing
        x_dict = gnn_model.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        x_dict = gnn_model.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # Extract 64-D Patient Node representations
        patient_embeddings = x_dict['patient'].cpu().numpy()

    # Reduce 64D -> 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000)
    embeddings_2d = tsne.fit_transform(patient_embeddings)

    # Plot 2D scatter
    plt.figure(figsize=(10, 8))
    palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
   
    for class_id, class_name in enumerate(ETIOLOGY_NAMES):
        mask = (y_test_labels == class_id)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=palette[class_id],
            label=class_name,
            alpha=0.8,
            edgecolors='k',
            s=50
        )

    plt.title('t-SNE Visualization of Hetero-GNN Patient Embeddings', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Clinical Etiology', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 3. Call the visualization function
visualize_gnn_embeddings_tsne(gnn_model, test_graph, y_test)import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. Execute pipeline and capture objects globally
gnn_model, test_graph, y_test, final_preds, target_names_with_grey = run_full_hybrid_pipeline(
    alpha=0.50,
    lower_bound=0.35,
    upper_bound=0.65
)

# 2. t-SNE Visualizer Function
def visualize_gnn_embeddings_tsne(gnn_model, test_graph, y_test_labels):
    gnn_model.eval()
   
    with torch.no_grad():
        x_dict = test_graph.x_dict
        edge_index_dict = test_graph.edge_index_dict

        # Forward projection through feature layers
        x_dict['patient'] = F.relu(gnn_model.patient_proj(x_dict['patient']))
        x_dict['biomarker'] = F.relu(gnn_model.biomarker_proj(x_dict['biomarker']))
        x_dict['etiology'] = F.relu(gnn_model.etiology_proj(x_dict['etiology']))

        # Heterogeneous Message Passing
        x_dict = gnn_model.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        x_dict = gnn_model.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # Extract 64-D Patient Node representations
        patient_embeddings = x_dict['patient'].cpu().numpy()

    # Reduce 64D -> 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000)
    embeddings_2d = tsne.fit_transform(patient_embeddings)

    # Plot 2D scatter
    plt.figure(figsize=(10, 8))
    palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
   
    for class_id, class_name in enumerate(ETIOLOGY_NAMES):
        mask = (y_test_labels == class_id)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=palette[class_id],
            label=class_name,
            alpha=0.8,
            edgecolors='k',
            s=50
        )

    plt.title('t-SNE Visualization of Hetero-GNN Patient Embeddings', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Clinical Etiology', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 3. Call the visualization function
visualize_gnn_embeddings_tsne(gnn_model, test_graph, y_test)