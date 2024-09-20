import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

from statsmodels.graphics.mosaicplot import mosaic
import itertools

from src.utils.network_utils import * 
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.options.mode.chained_assignment = None

def save_to_excel(filename, dataframes):
    with pd.ExcelWriter(filename) as writer:
        for name, df in dataframes:
            df.to_excel(writer, sheet_name=name)
            
def get_frequency_PS_LN(df, count, category):
    # Defining the columns related to positive lymph node counts (excluding sum of regions)
    pos_columns = [col for col in df.columns if col.startswith('pos_')]
    
    # Excluding the specified columns from the lymph node stations
    excluded_columns = ["pos_neckLN", "pos_mediaLN", "pos_abdoLN", 
                        "pos_neckLN_new", "pos_mediaLN_new", "pos_abdoLN_new", 
                        "pos_ON", "pos_OM", "pos_OA"]
    specific_pos_columns = [col for col in pos_columns if col not in excluded_columns]
    
    # Filtering the patients based on counts and N_category
    if count == None:
        metastasis_df = df[df['N_category'] == category]
    if category == None:
        metastasis_df = df[df['total_pos_LN'] <= count]

    # Creating a list to store the edges of the network for up to 2 metastasis
    edges = []

    # Iterating through the filtered patients and creating edges from primary site to lymph node stations
    for index, row in metastasis_df.iterrows():
        primary_site = row['Primary_Site']
        
        # Extracting the specific lymph node stations with positive counts
        nodes = [col[4:] for col in specific_pos_columns if pd.to_numeric(row[col], errors='coerce') > 0]
        
        # Creating edges from primary site to specific lymph node stations
        patient_edges = [(primary_site, node) for node in nodes]
        edges.extend(patient_edges)

    # Converting the edges to a DataFrame
    edges_df = pd.DataFrame(edges, columns=['Primary_Site', 'Node_Station'])

    # Counting the frequency of each edge, then sort by frequency
    edges_frequency = edges_df.groupby(['Primary_Site', 'Node_Station']).size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False)

    return edges_frequency

def combine_edge_frequencies_PS(descriptor):
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'all']
    dfs = []

    for num in numbers:
        df_name = f"EF_PS_{num}_{descriptor}"
        
        # Check if dataframe with the name exists in globals
        if df_name in globals():
            df = globals()[df_name]
            df = df.reset_index(drop=True)  # Reset the index
            df.columns = pd.MultiIndex.from_product([[num], df.columns])
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)
    return combined_df

def combine_edge_frequencies_PS_N_Category(descriptor):
    numbers = ['N1', 'N2', 'N3']
    dfs = []

    for num in numbers:
        df_name = f"EF_PS_{num}_{descriptor}"
        
        # Check if dataframe with the name exists in globals
        if df_name in globals():
            df = globals()[df_name]
            df = df.reset_index(drop=True)  # Reset the index
            df.columns = pd.MultiIndex.from_product([[num], df.columns])
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)
    return combined_df

def visualize_Network_PS_LN(edges_frequency, networkType):
    G = nx.Graph()
    for index, row in edges_frequency.iterrows():
        G.add_edge(row['Primary_Site'], row['Node_Station'], weight=row['Frequency'])
    
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    color_map = ['red' if node == 'lower' else 'blue' if node == 'mid' else 'green' if node == 'upper' else 'lightgray' for node in G.nodes()]
    plt.figure(figsize=(14, 14))
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=900, node_color=color_map, font_size=10, font_color="black", width=edge_widths)
    plt.title(f"{networkType} Network of Primary Sites to Lymphnode Stations", fontsize=20)
    plt.show()

def visualize_Network_PS_LN_subplot(edges_frequency, networkType, ax):
    G = nx.Graph()
    for index, row in edges_frequency.iterrows():
        G.add_edge(row['Primary_Site'], row['Node_Station'], weight=row['Frequency'])
    
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    color_map = ['red' if node == 'lower' else 'blue' if node == 'mid' else 'green' if node == 'upper' else 'lightgray' for node in G.nodes()]
    
    # Get all x and y coordinates
    x_coords = [x for x, y in Lymphnode_Positions.values()]
    y_coords = [y for x, y in Lymphnode_Positions.values()]
    
    # Find the min and max coordinates with some margin
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    # Draw the network
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=900, node_color=color_map, font_size=15, font_color="black", width=edge_widths, ax=ax)
    
    # Set title and axis limits
    ax.set_title(f"{networkType} Network", fontsize=18)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
def plot_PS_3x3_networks(descriptor):
    fig, axs = plt.subplots(3, 3, figsize=(24, 24))
    for i in range(1, 10):  # For metastasis counts from 1 to 9
        row = (i - 1) // 3
        col = (i - 1) % 3
        ax = axs[row, col]
        
        # Retrieve the global variable name for the edge frequency
        var_name = f"EF_PS_{i}_{descriptor}"
        
        # Retrieve the edge frequency from the global variable
        edges_frequency = globals().get(var_name, None)
        
        if edges_frequency is not None:
            visualize_Network_PS_LN_subplot(edges_frequency, f"{descriptor} <= {i}", ax)
    
    plt.suptitle(f"Networks for {descriptor}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # To ensure the title fits
    
    # Save the figure
    save_path = f"../Results/Network_Analysis_PS_{descriptor}.png"
    plt.savefig(save_path)
    
    plt.show()
    
def plot_PS_1x3_networks(descriptor):
    fig, axs = plt.subplots(1, 3, figsize=(24, 12))  # Change to 1x3 and adjust size
    for i in range(1, 4):  # For N_categories from 1 to 3
        ax = axs[i - 1]  # Index directly into axs for 1D array
        
        # Retrieve the global variable name for the edge frequency
        var_name = f"EF_PS_N{i}_{descriptor}"  # Assuming your edge frequencies for N_categories are named like this
        
        # Retrieve the edge frequency from the global variable
        edges_frequency = globals().get(var_name, None)
        
        if edges_frequency is not None:
            visualize_Network_PS_LN_subplot(edges_frequency, f"{descriptor} N{i}", ax)
    
    plt.suptitle(f"Networks for {descriptor} by N_category", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # To ensure the title fits
    
    # Save the figure
    save_path = f"../Results/Network_Analysis_PS_Ncategory_{descriptor}.png"
    plt.savefig(save_path)
    
    plt.show()
    
def size_category(proportion):
    thresholds = [0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20]
    sizes = [300, 500, 700, 1000, 1300, 1700, 2100, 2500, 3000, 3500]
    
    for t, s in zip(thresholds, sizes):
        if proportion < t:
            return s
    return sizes[-1] 
    
def visualize_Network_PS_LN_subplot_final(edges_frequency, networkType, ax):
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Dictionary holding the mapping between nodes and their respective colors for each network type
    color_mapping = {
            "T1_upper": {"101L": "#98BF64"},
            "T24_upper": {
                "102L": "#98BF64",
                "104R": "#98BF64",
                "104L": "#98BF64",
                "8": "#98BF64",
                "9": "#006400"
            },
            "T1_mid": {
                "102L": "#98BF64",
                "104R": "#98BF64",
                "112pulR": "#98BF64"
            },
            "T24_mid": {
                "102R": "#98BF64",
                "104L": "#98BF64",
                "102L": "#006400"
            },
            "T1_lower": {},
            "T24_lower": {
                "101L": "#98BF64",
                "101R": "#006400",
                "104R": "#006400",
                "104L": "#006400",
                "112pulR": "#006400"
            }
        }

    G = nx.Graph()
    node_counts = {}  # Dictionary to store counts of each node

    for index, row in edges_frequency.iterrows():
        if row['Node_Station'] not in G:
            G.add_node(row['Node_Station'])
            node_counts[row['Node_Station']] = 0

        # Update node counts based on frequency
        node_counts[row['Node_Station']] += row['Frequency']

    # Calculate the total count of all nodes
    total_count = sum(node_counts.values())

    # Determine the size of each node as a proportion of the total count multiplied by 100 for visualization
    node_sizes = [size_category(node_counts[node] / total_count) for node in G.nodes()]

    color_map = [color_mapping[networkType].get(node, '#FFD700') for node in G.nodes()]

    # Get all x and y coordinates
    x_coords = [x for x, y in Lymphnode_Positions.values()]
    y_coords = [y for x, y in Lymphnode_Positions.values()]

    # Find the min and max coordinates with some margin
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1

    # Draw the network without edges
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=node_sizes, node_color=color_map, font_size=14, font_color="black", ax=ax)

    # Set title and axis limits
    ax.set_title(f"{networkType} Network Plot", fontsize=18)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

# Modify the function to create a 2x3 grid of subplots for each set of metastasis counts (2 and 6)
def plot_PS_specific_networks(descriptors, count):
    fig, axs = plt.subplots(3, 2, figsize=(28, 42))  # Create a 2x3 grid of subplots
    ax_idx = 0  # Initialize the axis index to 0

    for descriptor in descriptors:
        ax = axs[ax_idx // 2, ax_idx % 2]  # Get the current axis
        edge_frequency_var = f"EF_PS_{count}_{descriptor}"

        # Check if the variable exists in the global namespace
        if edge_frequency_var in globals():
            edge_frequency_df = globals()[edge_frequency_var]
            visualize_Network_PS_LN_subplot_final(edge_frequency_df, descriptor, ax)

        ax_idx += 1  # Increment the axis index

    # Save the figure
    save_path = f"../Results/Network_Analysis_Figure1.svg"
    plt.savefig(save_path, format='svg', dpi=1200)  # Save as SVG with high DPI
    save_path_png = f"../Results/Network_Analysis_Figure1.png"
    plt.savefig(save_path_png, dpi=600)  # Save as PNG
    plt.show()

def get_frequency_TN(df, count, category):        
    # Defining the columns related to positive lymph node counts (excluding sum of regions)
    pos_columns = [col for col in df.columns if col.startswith('pos_')]
    
    # Excluding the specified columns from the lymph node stations
    excluded_columns = ["pos_neckLN", "pos_mediaLN", "pos_abdoLN", 
                        "pos_neckLN_new", "pos_mediaLN_new", "pos_abdoLN_new", 
                        "pos_ON", "pos_OM", "pos_OA"]
    specific_pos_columns = [col for col in pos_columns if col not in excluded_columns]
    
    # Filtering the patients based on counts and N_category
    if count == None:
        metastasis_df = df[df['N_category'] == category]
    if category == None:
        metastasis_df = df[df['total_pos_LN'] <= count]
        
    # Creating a list to store the edges of the network for up to 2 metastasis
    edges = []

    # Iterating through the filtered patients and creating edges from primary site to lymph node stations
    for index, row in metastasis_df.iterrows():        
        # Extracting the node stations with positive counts and ensuring the comparison is made with numeric values
        nodes = [col[4:] for col in specific_pos_columns if pd.to_numeric(row[col], errors='coerce') > 0]

        # Creating edges by connecting all the nodes within a patient
        patient_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
        edges.extend(patient_edges)

    # Converting the edges to a DataFrame
    edges_df = pd.DataFrame(edges, columns=['Node_1', 'Node_2'])

    # Counting the frequency of each edge, then sort values
    edges_frequency = edges_df.groupby(['Node_1', 'Node_2']).size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False)

    return edges_frequency

def combine_edge_frequencies(descriptor):
    numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'all']
    dfs = []

    for num in numbers:
        df_name = f"EF_TN_{num}_{descriptor}"
        
        # Check if dataframe with the name exists in globals
        if df_name in globals():
            df = globals()[df_name]
            df = df.reset_index(drop=True)  # Reset the index
            df.columns = pd.MultiIndex.from_product([[num], df.columns])
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)
    return combined_df

# Function to visualize the network
def visualize_Network_TN(edges_frequency, networkType):
    G = nx.Graph()

    for index, row in edges_frequency.iterrows():
        G.add_edge(row['Node_1'], row['Node_2'], weight=row['Frequency'])
    
    # Calculate unweighted degree of nodes
    node_degrees_df = identify_node_degrees(edges_frequency)
    node_size_map = dict(zip(node_degrees_df['Node'], node_degrees_df['Weighted Degree']))
    
    # Identify hub nodes (Top 5 nodes based on Weighted Degree)
    hub_nodes = node_degrees_df.head(5)['Node'].tolist()
    
    # Prepare color and size maps
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    color_map = ['gold' if node in hub_nodes else 'red' if node == 'abdoLN' else 'blue' if node == 'mediaLN' else 'green' if node == 'neckLN' else 'lightgray' for node in G.nodes()]
    size_map = [node_size_map.get(node, 1) * 10000 for node in G.nodes()]

    plt.figure(figsize=(14, 14))
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=size_map, node_color=color_map, font_size=15, font_color="black", width=edge_widths)
    plt.title(f"{networkType} Network of Two Lymphnode Stations", fontsize=20)
    plt.show()
    
def visualize_Network_TN_LN_subplot(edges_frequency, networkType, ax):
        # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    G = nx.Graph()
    for index, row in edges_frequency.iterrows():
        G.add_edge(row['Node_1'], row['Node_2'], weight=row['Frequency'])
    
    # Calculate unweighted degree of nodes
    node_degrees_df = identify_node_degrees(edges_frequency)
    node_size_map = dict(zip(node_degrees_df['Node'], node_degrees_df['Weighted Degree']))
    
    # Identify hub nodes (Top 5 nodes based on Weighted Degree)
    hub_nodes = node_degrees_df.head(5)['Node'].tolist()
    
    # Prepare color and size maps
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    color_map = ['gold' if node in hub_nodes else 'red' if node == 'abdoLN' else 'blue' if node == 'mediaLN' else 'green' if node == 'neckLN' else 'lightgray' for node in G.nodes()]
    size_map = [node_size_map.get(node, 1) * 5000 for node in G.nodes()]

    # Get all x and y coordinates
    x_coords = [x for x, y in Lymphnode_Positions.values()]
    y_coords = [y for x, y in Lymphnode_Positions.values()]
    
    # Find the min and max coordinates with some margin
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    # Draw the network
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=size_map, node_color=color_map, font_size=15, font_color="black", width=edge_widths, ax=ax)
    
    # Set title and axis limits
    ax.set_title(f"{networkType} Network", fontsize=18)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

def plot_TN_3x3_networks(descriptor):
    fig, axs = plt.subplots(3, 3, figsize=(42, 42))
    for i in range(1, 10):  # For metastasis counts from 2 to 10
        row = (i - 1) // 3
        col = (i - 1) % 3
        ax = axs[row, col]
        
        # Retrieve the global variable name for the edge frequency
        var_name = f"EF_TN_{i+1}_{descriptor}"
        
        # Retrieve the edge frequency from the global variable
        edges_frequency = globals().get(var_name, None)
        
        if edges_frequency is not None:
            visualize_Network_TN_LN_subplot(edges_frequency, f"{descriptor} <= {i+1}", ax)
    
    plt.suptitle(f"Networks for {descriptor}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # To ensure the title fits
    
    # Save the figure
    save_path = f"../Results/Network_Analysis_TN_{descriptor}.png"
    plt.savefig(save_path)
    
    plt.show()
    
def plot_TN_1x3_networks(descriptor):
    fig, axs = plt.subplots(1, 3, figsize=(42, 14))  # Change to 1x3 and adjust size
    for i in range(1, 4):  # For N_categories from 1 to 3
        ax = axs[i - 1]  # Index directly into axs for 1D array
        
        # Retrieve the global variable name for the edge frequency
        var_name = f"EF_TN_N{i}_{descriptor}"  # Assuming your edge frequencies for N_categories are named like this
        
        # Retrieve the edge frequency from the global variable
        edges_frequency = globals().get(var_name, None)
        
        if edges_frequency is not None:
            visualize_Network_TN_LN_subplot(edges_frequency, f"{descriptor} N{i}", ax)
    
    plt.suptitle(f"Networks for {descriptor} by N_category", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # To ensure the title fits
    
    # Save the figure
    save_path = f"../Results/Network_Analysis_TN_Ncategory_{descriptor}.png"
    plt.savefig(save_path)
    
    plt.show()
    
# Modify the function to create a 2x3 grid of subplots for each set of metastasis counts (2 and 6)
def plot_TN_specific_networks(descriptors, count):
        # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    fig, axs = plt.subplots(3, 2, figsize=(28, 42))  # Create a 2x3 grid of subplots
    ax_idx = 0  # Initialize the axis index to 0

    for descriptor in descriptors:
        ax = axs[ax_idx // 2, ax_idx % 2]  # Get the current axis
        edge_frequency_var = f"EF_TN_{count}_{descriptor}"

        # Check if the variable exists in the global namespace
        if edge_frequency_var in globals():
            edge_frequency_df = globals()[edge_frequency_var]

            # Visualize the network
            if count == 2:
                visualize_Network_TN_LN_subplot(edge_frequency_df, f"{descriptor} at N1 metastasis", ax)
            elif count == 6:
                visualize_Network_TN_LN_subplot(edge_frequency_df, f"{descriptor} at N1-2 metastasis", ax)
            else:
                visualize_Network_TN_LN_subplot(edge_frequency_df, f"{descriptor} <= {count} metastasis", ax)

        ax_idx += 1  # Increment the axis index

    # plt.tight_layout()  # Adjust the layout to avoid overlaps

    # Save the figure (svg)
    save_path = f"../Results/Network_Analysis_TN_Main_Figures_{count}.svg"
    plt.savefig(save_path, format='svg', dpi=1200)  # Save as SVG with high DPI
    # Save the figure (png)
    save_path_png = f"../Results/Network_Analysis_TN_Main_Figures_{count}.png"
    plt.savefig(save_path_png)
    plt.show()
    
# Function to identify hub nodes based on both unweighted degree and normalized strength
def identify_node_degrees(edge_frequency):
    # Initialize an empty graph
    G = nx.Graph()
    
    # Populate the graph based on edges and their frequencies
    for index, row in edge_frequency.iterrows():
        G.add_edge(row['Node_1'], row['Node_2'], weight=row['Frequency'])
    
    # Calculate the unweighted degree of each node
    node_degrees = dict(G.degree())
    
    # Calculate the total frequency (sum of all edge weights)
    total_frequency = sum(weight for _, _, weight in G.edges(data='weight'))
    
    # Calculate the normalized strength of each node
    node_normalized_strengths = {node: round(sum(weight / total_frequency for _, _, weight in G.edges(node, data='weight')), 3) for node in G.nodes()}
 
    # Create a DataFrame to store the nodes, their degrees, and normalized strengths
    hub_nodes_df = pd.DataFrame({
        'Node': list(node_degrees.keys()),
        'Unweighted Degree': list(node_degrees.values()),
        'Weighted Degree': [node_normalized_strengths.get(node, 0) for node in node_degrees.keys()]
    })
    
    # Sort the DataFrame based on both Weighted Degree
    hub_nodes_df.sort_values(by=['Weighted Degree'], ascending=[False], inplace=True)
    
    return hub_nodes_df

# Function to combine all the identified node degrees based on different criteria
def combine_node_degrees(descriptor):
    numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    dfs = []

    for num in numbers:
        df_name = f"EF_TN_{num}_{descriptor}"
        
        # Check if dataframe with the name exists in globals
        if df_name in globals():
            df = globals()[df_name]
            node_degrees_df = identify_node_degrees(df)  # Identify the node degrees for the current DataFrame
            node_degrees_df = node_degrees_df.reset_index(drop=True)  # Reset the index
            
            # Rename the columns
            node_degrees_df.rename(columns={'Unweighted Degree': 'UW Degree', 'Weighted Degree': 'W Degree'}, inplace=True)
            
            node_degrees_df.columns = pd.MultiIndex.from_product([[num], node_degrees_df.columns])
            dfs.append(node_degrees_df)

    combined_df = pd.concat(dfs, axis=1)
    return combined_df

def node_sizes(proportion):
    thresholds = [0.03, 0.06, 0.1, 0.14, 0.18, 0.22, 0.27, 0.32, 0.37, 0.41, 0.45]
    sizes = [300, 500, 700, 1000, 1400, 1700, 2100, 2500, 3000, 3600]
    
    for t, s in zip(thresholds, sizes):
        if proportion < t:
            return s
    return sizes[-1] 
    
def visualize_Network_TN_LN_subplot_final(edges_frequency, networkType, ax, color_mapping):
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    G = nx.Graph()
    for index, row in edges_frequency.iterrows():
        G.add_edge(row['Node_1'], row['Node_2'], weight=row['Frequency'])
    
    # Calculate unweighted degree of nodes
    node_degrees_df = identify_node_degrees(edges_frequency)
    node_size_map = dict(zip(node_degrees_df['Node'], node_degrees_df['Weighted Degree']))

    # Adjust the node sizes based on the size_category function
    for node, degree in node_size_map.items():
        node_size_map[node] = node_sizes(degree)
    size_map = [node_size_map.get(node, 1) for node in G.nodes()]

    hub_nodes = node_degrees_df.head(5)['Node'].tolist()
    
    # Prepare color and size maps
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    color_map = [color_mapping[networkType].get(node, 'lightgray') for node in G.nodes()]

    # Get all x and y coordinates
    x_coords = [x for x, y in Lymphnode_Positions.values()]
    y_coords = [y for x, y in Lymphnode_Positions.values()]
    
    # Find the min and max coordinates with some margin
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    # Draw the network
    nx.draw(G, Lymphnode_Positions, with_labels=True, node_size=size_map, node_color=color_map, font_size=15, font_color="black", width=edge_widths, ax=ax)
    
    # Set title and axis limits
    ax.set_title(f"{networkType} Network", fontsize=18)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

# Modify the function to create a 2x3 grid of subplots for each set of metastasis counts (all)
def plot_TN_specific_networks_final(descriptors, count, color_mapping):
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    fig, axs = plt.subplots(3, 2, figsize=(28, 42))  # Create a 2x3 grid of subplots
    ax_idx = 0  # Initialize the axis index to 0

    for descriptor in descriptors:
        ax = axs[ax_idx // 2, ax_idx % 2]  # Get the current axis
        edge_frequency_var = f"EF_TN_{count}_{descriptor}"

        # Check if the variable exists in the global namespace
        if edge_frequency_var in globals():
            edge_frequency_df = globals()[edge_frequency_var]

            # Visualize the network
            visualize_Network_TN_LN_subplot_final(edge_frequency_df, descriptor, ax, color_mapping)

        ax_idx += 1  # Increment the axis index

    # Save the figure
    save_path = f"../Results/Network_Analysis_Figure3.svg"
    plt.savefig(save_path, format='svg', dpi=1200)  # Save as SVG with high DPI
    
    save_path_png = f"../Results/Network_Analysis_Figure3.png"
    plt.savefig(save_path_png, dpi=600)  # Save as SVG with high DPI
    plt.show()
    
#Label the Node_Station entries based on their frequencies without accumulating the frequencies.
def group_by_frequencies(data):
    # Mapping for the frequencies
    freq_map = {
        1: 'Freq=1',
        2: 'Freq=2',
        3: 'Freq=3',
        4: 'Freq=4',
        5: 'Freq=5'
    }
    
    # Modify Node_Station based on the frequency
    mask = data['Frequency'].isin([1, 2, 3, 4, 5])
    data.loc[mask, 'Node_Station'] = data.loc[mask, 'Frequency'].map(freq_map)
    
    return data

def filter_low_frequencies(df, threshold):
    return df[df['Frequency'] >= threshold]

def plot_mosaic(df, title):
    mosaic_data = df.set_index(['Primary_Site', 'Node_Station'])['Frequency']
    plt.figure(figsize=(20, 30))
    mosaic(mosaic_data, title=title)
    plt.show()

def plot_nested_pie(df, counts, Type):
    # Filtering the patients with up to the specified total number of positive lymph nodes detected
    metastasis_df = df[df['total_pos_LN'] <= counts]
    
    # Grouping the data by 'Primary_Site' and then summing the positive counts for each lymph node location
    grouped_df = metastasis_df.groupby('Primary_Site')[['pos_neckLN', 'pos_mediaLN', 'pos_abdoLN']].sum().reset_index()

    # Extracting data for the pie charts
    primary_sites = grouped_df['Primary_Site'].str.upper().tolist()
    neckLN_counts = grouped_df['pos_neckLN'].tolist()
    mediaLN_counts = grouped_df['pos_mediaLN'].tolist()
    abdoLN_counts = grouped_df['pos_abdoLN'].tolist()

    # Data for outer pie chart
    outer_sizes = grouped_df[['pos_neckLN', 'pos_mediaLN', 'pos_abdoLN']].sum(axis=1).tolist()
    total_outer = sum(outer_sizes)

    # Data for inner pie chart
    inner_sizes = []
    for i in range(len(primary_sites)):
        inner_sizes.extend([neckLN_counts[i], mediaLN_counts[i], abdoLN_counts[i]])

    # Colors for the charts
    outer_colors = ['darkred', 'darkgreen', 'darkblue']
    inner_colors = ['red', 'orangered', 'pink', 'olive', 'yellowgreen', 'lime', 'blue', 'lightblue', 'aqua']

    # Calculating the labels for the inner and outer pie charts with percentages
    outer_labels_adjusted = []
    inner_labels_adjusted = []
    for i in range(len(primary_sites)):
        outer_percentage = (outer_sizes[i] / total_outer) * 100
        outer_labels_adjusted.append(f"{primary_sites[i]} ({outer_percentage:.1f}%)")
        
        for count, label in zip([neckLN_counts[i], mediaLN_counts[i], abdoLN_counts[i]], ["NeckLN", "MediaLN", "AbdoLN"]):
            inner_percentage = (count / outer_sizes[i]) * 100
            if count > 0:
                inner_labels_adjusted.append(f"{label} ({inner_percentage:.1f}%)")
    
    # Creating the adjusted nested pie chart with labels inside the pies
    fig, ax = plt.subplots(figsize=(12, 12))

    # Outer pie with adjusted label positions
    wedges, _ = ax.pie(outer_sizes, labels=None, colors=outer_colors, radius=1.3, wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # Place the labels in the center of the outer pie slices
    for i, wedge in enumerate(wedges):
        y = (wedge.theta2 + wedge.theta1) / 2.
        x = (1 + wedge.r) / 2. * np.cos(np.deg2rad(y))
        y = (1 + wedge.r) / 2. * np.sin(np.deg2rad(y))
        plt.text(x, y, outer_labels_adjusted[i], ha='center', va='center', fontsize=10, fontweight='bold')

    # Inner pie with adjusted label positions
    wedges_inner, texts_inner = ax.pie(inner_sizes, labels=inner_labels_adjusted, colors=inner_colors, radius=1.0, wedgeprops=dict(width=0.3, edgecolor='w'), labeldistance=0.7)
    
    # Adjust the position of inner pie labels
    for i, wedge in enumerate(wedges_inner):
        y = (wedge.theta2 + wedge.theta1) / 2.
        x = 0.7 * (1 + wedge.r) / 2. * np.cos(np.deg2rad(y))
        y = 0.7 * (1 + wedge.r) / 2. * np.sin(np.deg2rad(y))
        texts_inner[i].set_position((x, y))
    
    # Increase font size and bold the labels of the inner pie
    for text in texts_inner:
        text.set_size(10)
        text.set_weight('bold')   

    # Adjust title position
    plt.title(f"{Type} Distribution of Lymph Node Locations Metastasis", fontsize=20, y=1.1)
    
    plt.show()