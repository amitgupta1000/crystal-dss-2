"""
Standalone module to build causality network graph from saved Granger test results.

This allows rebuilding the adjacency matrix and network graph with different parameters
without re-running expensive Granger causality tests.

Usage:
    python backend/build_causality_graph.py
    
Or with custom parameters:
    python backend/build_causality_graph.py --alpha 0.05 --min-connections 3
"""

import argparse
import logging
import sys
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: networkx and/or matplotlib not installed. Visualization will not be available.")
    print("Install with: pip install networkx matplotlib")

from src.file_utils import download_latest_csv_from_gcs, save_dataframe_to_gcs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# HELPER FUNCTIONS FOR ADJACENCY MATRIX AND NETWORK GRAPH BUILDING
# ============================================================================

def build_adjacency_matrix_from_results(
    granger_results_df: pd.DataFrame,
    alpha: float = 0.04
) -> pd.DataFrame:
    """
    Build causality adjacency matrix from saved Granger test results.
    
    This allows rebuilding the matrix with different alpha thresholds
    without re-running expensive Granger causality tests.
    
    Args:
        granger_results_df: DataFrame with columns ['Source', 'Target', 'Lag', 'P-value (F-test)']
        alpha: Significance threshold (default 0.04)
        
    Returns:
        pd.DataFrame: Adjacency matrix where [i,j]=1 means Source[i] Granger-causes Target[j]
    """
    if granger_results_df.empty:
        logger.warning("Empty Granger results - returning empty adjacency matrix")
        return pd.DataFrame()
    
    # Get unique commodities
    all_commodities = sorted(set(granger_results_df['Source'].unique()) | 
                            set(granger_results_df['Target'].unique()))
    
    # Initialize matrix with zeros
    adjacency_matrix = pd.DataFrame(
        0,
        index=all_commodities,
        columns=all_commodities
    )
    
    # Filter significant relationships (any lag below alpha threshold)
    significant = granger_results_df[granger_results_df['P-value (F-test)'] < alpha]
    
    # Mark significant relationships in matrix
    for _, row in significant.iterrows():
        source = row['Source']
        target = row['Target']
        adjacency_matrix.loc[source, target] = 1
    
    num_relationships = adjacency_matrix.sum().sum()
    logger.info("Built adjacency matrix: %d commodities, %d significant relationships (alpha=%.3f)",
               len(all_commodities), int(num_relationships), alpha)
    
    return adjacency_matrix


def build_causality_network_graph(
    granger_results_df: pd.DataFrame,
    adjacency_matrix_df: pd.DataFrame,
    alpha: float = 0.04,
    min_connection_count: int = 1
):
    """
    Build a network graph from Granger causality results for visualization.
    
    Args:
        granger_results_df: DataFrame with columns ['Source', 'Target', 'Lag', 'P-value (F-test)']
        adjacency_matrix_df: Adjacency matrix showing which commodities cause others
        alpha: Significance threshold for including edges
        min_connection_count: Minimum number of connections for a node to be included
        
    Returns:
        dict: Network graph in format suitable for D3.js or networkx visualization
    """
    # Filter significant relationships
    significant_df = granger_results_df[granger_results_df['P-value (F-test)'] < alpha].copy()
    
    # Get the most significant lag for each source-target pair
    best_lags = significant_df.loc[
        significant_df.groupby(['Source', 'Target'])['P-value (F-test)'].idxmin()
    ].copy()
    
    # Build nodes list
    all_commodities = set(best_lags['Source'].unique()) | set(best_lags['Target'].unique())
    
    # Calculate node metrics
    node_metrics = {}
    for commodity in all_commodities:
        outgoing = len(best_lags[best_lags['Source'] == commodity])
        incoming = len(best_lags[best_lags['Target'] == commodity])
        node_metrics[commodity] = {
            'id': commodity,
            'outgoing_connections': outgoing,
            'incoming_connections': incoming,
            'total_connections': outgoing + incoming,
            'net_influence': outgoing - incoming  # Positive means more causal influence
        }
    
    # Filter nodes by minimum connection count
    nodes = [
        metrics for metrics in node_metrics.values()
        if metrics['total_connections'] >= min_connection_count
    ]
    
    # Get set of included node IDs
    included_nodes = {node['id'] for node in nodes}
    
    # Build edges list (only for included nodes)
    edges = []
    for _, row in best_lags.iterrows():
        source = row['Source']
        target = row['Target']
        
        if source in included_nodes and target in included_nodes:
            edges.append({
                'source': source,
                'target': target,
                'lag': int(row['Lag']),
                'p_value': float(row['P-value (F-test)']),
                'weight': 1 - float(row['P-value (F-test)'])  # Convert p-value to weight (lower p = higher weight)
            })
    
    # Build graph structure
    network_graph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'alpha': alpha,
            'min_connection_count': min_connection_count,
            'total_commodities': len(all_commodities),
            'included_commodities': len(nodes),
            'significant_relationships': len(edges)
        }
    }
    
    logger.info("Built causality network graph: %d nodes, %d edges", len(nodes), len(edges))
    
    return network_graph


def save_causality_network_data(
    network_graph: dict,
    bucket_name: str,
    gcs_prefix: str
):
    """
    Save causality network graph data to GCS as JSON for frontend visualization.
    
    Args:
        network_graph: Network graph dictionary from build_causality_network_graph()
        bucket_name: GCS bucket name
        gcs_prefix: GCS path prefix (e.g., 'stats_studies_data/causality/network_graph.json')
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_prefix)
        
        # Convert to JSON
        json_data = json.dumps(network_graph, indent=2)
        
        # Upload to GCS
        blob.upload_from_string(json_data, content_type='application/json')
        
        logger.info("Saved causality network graph to GCS: gs://%s/%s", bucket_name, gcs_prefix)
        return f"gs://{bucket_name}/{gcs_prefix}"
        
    except Exception as e:
        logger.error("Failed to save causality network graph: %s", str(e))
        return None


def visualize_network_graph(network_graph: dict, output_file: str = None, figsize=(20, 16)):
    """
    Visualize causality network graph using networkx and matplotlib.
    
    Args:
        network_graph: Network graph dictionary
        output_file: Path to save visualization (e.g., 'causality_graph.png'). If None, displays interactively.
        figsize: Figure size in inches (width, height)
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization libraries not available. Install with: pip install networkx matplotlib")
        return
    
    logger.info("Creating network visualization...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in network_graph['nodes']:
        G.add_node(
            node['id'],
            outgoing=node['outgoing_connections'],
            incoming=node['incoming_connections'],
            net_influence=node['net_influence']
        )
    
    # Add edges with attributes
    for edge in network_graph['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            lag=edge['lag'],
            p_value=edge['p_value'],
            weight=edge['weight']
        )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout - use spring layout for better separation
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on total connections
    node_sizes = [G.nodes[node].get('outgoing', 0) + G.nodes[node].get('incoming', 0) for node in G.nodes()]
    node_sizes = [size * 100 + 300 for size in node_sizes]  # Scale for visibility
    
    # Node colors based on net influence (red = high influence, blue = low influence)
    net_influences = [G.nodes[node].get('net_influence', 0) for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=net_influences,
        cmap=plt.cm.RdYlBu_r,
        alpha=0.8,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        width=1,
        ax=ax,
        connectionstyle='arc3,rad=0.1'
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Add title and metadata
    metadata = network_graph['metadata']
    title = f"Causality Network Graph\n"
    title += f"Alpha: {metadata['alpha']:.3f} | "
    title += f"Nodes: {metadata['included_commodities']} | "
    title += f"Edges: {metadata['significant_relationships']}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlBu_r,
        norm=plt.Normalize(vmin=min(net_influences), vmax=max(net_influences))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Net Influence (Outgoing - Incoming)', rotation=270, labelpad=20)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info("✓ Saved visualization to: %s", output_file)
        plt.close()
    else:
        plt.show()


def create_interactive_html(network_graph: dict, output_file: str = 'causality_graph.html'):
    """
    Create an interactive HTML visualization using vis.js.
    
    Args:
        network_graph: Network graph dictionary
        output_file: Path to save HTML file
    """
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Causality Network Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #mynetwork { width: 100%; height: 800px; border: 1px solid lightgray; }
        #info { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .metadata { margin: 10px 0; }
        .legend { display: flex; gap: 20px; margin-top: 15px; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 20px; height: 20px; border-radius: 50%; }
    </style>
</head>
<body>
    <h1>Causality Network Graph</h1>
    <div id="info">
        <div class="metadata">
            <strong>Alpha:</strong> {{ALPHA}} | 
            <strong>Nodes:</strong> {{NODES}} | 
            <strong>Edges:</strong> {{EDGES}} | 
            <strong>Total Commodities:</strong> {{TOTAL}}
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>High Influence (many outgoing)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db;"></div>
                <span>Low Influence (many incoming)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #95a5a6;"></div>
                <span>Neutral</span>
            </div>
        </div>
    </div>
    <div id="mynetwork"></div>
    
    <script type="text/javascript">
        var graphData = {{GRAPH_DATA}};
        
        // Prepare nodes for vis.js
        var nodes = graphData.nodes.map(function(node) {
            var netInfluence = node.net_influence;
            var maxInfluence = Math.max(...graphData.nodes.map(n => Math.abs(n.net_influence)));
            var normalizedInfluence = netInfluence / (maxInfluence || 1);
            
            // Color scale: red for high influence, blue for low/negative
            var color;
            if (normalizedInfluence > 0.5) {
                color = '#e74c3c'; // Red
            } else if (normalizedInfluence < -0.5) {
                color = '#3498db'; // Blue
            } else {
                color = '#95a5a6'; // Gray
            }
            
            return {
                id: node.id,
                label: node.id,
                value: node.total_connections,
                color: color,
                title: 'Outgoing: ' + node.outgoing_connections + '\\nIncoming: ' + node.incoming_connections + '\\nNet Influence: ' + node.net_influence
            };
        });
        
        // Prepare edges for vis.js
        var edges = graphData.edges.map(function(edge) {
            return {
                from: edge.source,
                to: edge.target,
                arrows: 'to',
                label: 'lag ' + edge.lag,
                title: 'P-value: ' + edge.p_value.toFixed(4) + '\\nLag: ' + edge.lag,
                color: { opacity: edge.weight }
            };
        });
        
        var container = document.getElementById('mynetwork');
        var data = { nodes: nodes, edges: edges };
        var options = {
            nodes: {
                shape: 'dot',
                font: { size: 14, color: '#000' },
                borderWidth: 2,
                borderWidthSelected: 4
            },
            edges: {
                font: { size: 10, align: 'middle' },
                smooth: { type: 'cubicBezier', roundness: 0.3 }
            },
            physics: {
                stabilization: { iterations: 200 },
                barnesHut: {
                    gravitationalConstant: -8000,
                    springLength: 200,
                    springConstant: 0.04
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true
            }
        };
        
        var network = new vis.Network(container, data, options);
        
        // Click event
        network.on('selectNode', function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var node = graphData.nodes.find(n => n.id === nodeId);
                alert(nodeId + ':\\n\\n' +
                      'Outgoing: ' + node.outgoing_connections + '\\n' +
                      'Incoming: ' + node.incoming_connections + '\\n' +
                      'Net Influence: ' + node.net_influence);
            }
        });
    </script>
</body>
</html>"""
    
    # Replace placeholders
    html_content = html_template.replace('{{GRAPH_DATA}}', json.dumps(network_graph))
    html_content = html_content.replace('{{ALPHA}}', str(network_graph['metadata']['alpha']))
    html_content = html_content.replace('{{NODES}}', str(network_graph['metadata']['included_commodities']))
    html_content = html_content.replace('{{EDGES}}', str(network_graph['metadata']['significant_relationships']))
    html_content = html_content.replace('{{TOTAL}}', str(network_graph['metadata']['total_commodities']))
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("✓ Created interactive HTML visualization: %s", output_file)
    logger.info("  Open this file in your web browser to explore the network interactively")


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Build causality network graph from saved Granger test results'
    )
    parser.add_argument(
        '--bucket',
        default='crystal-dss',
        help='GCS bucket name (default: crystal-dss)'
    )
    parser.add_argument(
        '--granger-results-prefix',
        default='stats_studies_data/causality/all_granger_test_results.csv',
        help='GCS prefix for Granger test results CSV'
    )
    parser.add_argument(
        '--output-prefix',
        default='stats_studies_data/causality',
        help='GCS prefix for output files'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.04,
        help='Significance threshold for causality relationships (default: 0.04)'
    )
    parser.add_argument(
        '--min-connections',
        type=int,
        default=2,
        help='Minimum connections for a node to be included in graph (default: 2)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create matplotlib visualization (PNG)'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Create interactive HTML visualization'
    )
    parser.add_argument(
        '--output-viz',
        default='causality_graph.png',
        help='Output filename for visualization (default: causality_graph.png)'
    )
    parser.add_argument(
        '--output-html',
        default='causality_graph.html',
        help='Output filename for HTML visualization (default: causality_graph.html)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show detailed debugging information'
    )
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    bucket_name = args.bucket
    granger_results_prefix = args.granger_results_prefix
    output_prefix = args.output_prefix
    alpha = args.alpha
    min_connection_count = args.min_connections
    
    logger.info("="*80)
    logger.info("CAUSALITY NETWORK GRAPH BUILDER")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info("  Bucket: %s", bucket_name)
    logger.info("  Granger results: %s", granger_results_prefix)
    logger.info("  Output prefix: %s", output_prefix)
    logger.info("  Alpha (significance): %.4f", alpha)
    logger.info("  Min connections per node: %d", min_connection_count)
    
    # Step 1: Load Granger test results from GCS
    logger.info("\n" + "="*80)
    logger.info("LOADING GRANGER TEST RESULTS FROM GCS")
    logger.info("="*80)
    
    try:
        # Download the specific Granger test results file from GCS
        # We need to be explicit about the file name, not just the folder
        from google.cloud import storage
        import io
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Construct the exact blob path
        blob_path = 'stats_studies_data/causality/all_granger_test_results.csv'
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            logger.error("Granger test results file not found at: gs://%s/%s", bucket_name, blob_path)
            logger.error("Make sure you have run dss_analyst.py first to generate the test results.")
            sys.exit(1)
        
        # Download and load the CSV
        csv_data = blob.download_as_string()
        granger_results_df = pd.read_csv(io.BytesIO(csv_data))
        
        # Clean column names (strip whitespace)
        granger_results_df.columns = granger_results_df.columns.str.strip()
        
        logger.info("✓ Loaded Granger test results from: gs://%s/%s", bucket_name, blob_path)
        logger.info("  Shape: %s", granger_results_df.shape)
        logger.info("  Columns: %s", list(granger_results_df.columns))
        
        if args.debug:
            logger.debug("First few rows:")
            logger.debug("\n%s", granger_results_df.head())
        
        # Validate required columns
        required_cols = ['Source', 'Target', 'Lag', 'P-value (F-test)']
        missing_cols = [col for col in required_cols if col not in granger_results_df.columns]
        if missing_cols:
            logger.error("Missing required columns: %s", missing_cols)
            logger.error("Available columns: %s", list(granger_results_df.columns))
            logger.error("Make sure you have run dss_analyst.py first to generate the test results.")
            sys.exit(1)
        
        logger.info("  Unique sources: %d", granger_results_df['Source'].nunique())
        logger.info("  Unique targets: %d", granger_results_df['Target'].nunique())
        logger.info("  Total test results: %d", len(granger_results_df))
        
    except Exception as e:
        logger.error("Failed to load Granger test results: %s", e)
        logger.error("Make sure you have run dss_analyst.py first to generate the test results.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Build adjacency matrix
    logger.info("\n" + "="*80)
    logger.info("BUILDING ADJACENCY MATRIX")
    logger.info("="*80)
    
    try:
        adjacency_matrix_df = build_adjacency_matrix_from_results(
            granger_results_df=granger_results_df,
            alpha=alpha
        )
        
        if adjacency_matrix_df.empty:
            logger.warning("Empty adjacency matrix - no significant relationships found")
        else:
            num_relationships = adjacency_matrix_df.sum().sum()
            logger.info("✓ Adjacency matrix built")
            logger.info("  Size: %dx%d", adjacency_matrix_df.shape[0], adjacency_matrix_df.shape[1])
            logger.info("  Significant relationships: %d", int(num_relationships))
            
            # Save adjacency matrix
            adjacency_prefix = f'{output_prefix}/causality_adjacency_matrix_alpha{alpha:.3f}.csv'
            save_dataframe_to_gcs(
                df=adjacency_matrix_df,
                bucket_name=bucket_name,
                gcs_prefix=adjacency_prefix,
                validate_rows=False,
                include_index=True
            )
            logger.info("✓ Saved adjacency matrix to: gs://%s/%s", bucket_name, adjacency_prefix)
            
    except Exception as e:
        logger.error("Failed to build adjacency matrix: %s", e)
        sys.exit(1)
    
    # Step 3: Build network graph
    logger.info("\n" + "="*80)
    logger.info("BUILDING NETWORK GRAPH")
    logger.info("="*80)
    
    try:
        network_graph = build_causality_network_graph(
            granger_results_df=granger_results_df,
            adjacency_matrix_df=adjacency_matrix_df,
            alpha=alpha,
            min_connection_count=min_connection_count
        )
        
        logger.info("✓ Network graph built")
        logger.info("  Nodes: %d", len(network_graph['nodes']))
        logger.info("  Edges: %d", len(network_graph['edges']))
        logger.info("  Total commodities: %d", network_graph['metadata']['total_commodities'])
        logger.info("  Included commodities: %d", network_graph['metadata']['included_commodities'])
        
        # Display top influencers
        if network_graph['nodes']:
            sorted_nodes = sorted(
                network_graph['nodes'],
                key=lambda x: x['net_influence'],
                reverse=True
            )
            logger.info("\n  Top 5 influencers (highest net influence):")
            for node in sorted_nodes[:5]:
                logger.info("    %s: net_influence=%d (out=%d, in=%d)",
                           node['id'],
                           node['net_influence'],
                           node['outgoing_connections'],
                           node['incoming_connections'])
        
        # Save network graph
        network_prefix = f'{output_prefix}/causality_network_graph_alpha{alpha:.3f}_min{min_connection_count}.json'
        result = save_causality_network_data(
            network_graph=network_graph,
            bucket_name=bucket_name,
            gcs_prefix=network_prefix
        )
        
        if result:
            logger.info("✓ Saved network graph to: %s", result)
        else:
            logger.warning("Failed to save network graph")
            
    except Exception as e:
        logger.error("Failed to build network graph: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("CAUSALITY NETWORK GRAPH GENERATION COMPLETE")
    logger.info("="*80)
    logger.info("\nOutputs saved to gs://%s/%s/", bucket_name, output_prefix)
    logger.info("  - causality_adjacency_matrix_alpha%.3f.csv", alpha)
    logger.info("  - causality_network_graph_alpha%.3f_min%d.json", alpha, min_connection_count)
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("\n" + "="*80)
        logger.info("CREATING MATPLOTLIB VISUALIZATION")
        logger.info("="*80)
        visualize_network_graph(network_graph, output_file=args.output_viz)
    
    if args.html:
        logger.info("\n" + "="*80)
        logger.info("CREATING INTERACTIVE HTML VISUALIZATION")
        logger.info("="*80)
        create_interactive_html(network_graph, output_file=args.output_html)


if __name__ == '__main__':
    main()
