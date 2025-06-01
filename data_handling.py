import pandas as pd
from lifelines.datasets import load_lung
import numpy as np

# --- Data Loading ---

def load_ncctg_lung_data():
    """Loads and preprocesses the NCCTG Lung Cancer dataset."""
    df = load_lung()
    df = df[['time', 'status']]
    df.columns = ['duration', 'event']
    df['event'] = df['event'].astype(int) # Ensure event is integer (0 or 1)
    return df

# --- Data Partitioning Strategies ---

class DataPartitioner:
    """Partitions a DataFrame into several smaller DataFrames based on different strategies."""
    def __init__(self, df):
        self.df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    def uniform_partition(self, num_nodes):
        """Partitions data uniformly among nodes."""
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive.")
        
        # Using np.array_split is a robust way to ensure all data is distributed uniformly.
        if len(self.df) == 0:
            return [pd.DataFrame(columns=self.df.columns) for _ in range(num_nodes)]
            
        indices = np.array_split(np.arange(len(self.df)), num_nodes)
        partitions = [self.df.iloc[idx_group].reset_index(drop=True) for idx_group in indices]
        return partitions

    def nonuniform_partition(self, num_nodes, fraction_major=0.5):
        """Partitions data non-uniformly, with one node having a larger fraction."""
        if not (0 < fraction_major < 1):
            raise ValueError("Fraction for major node must be between 0 and 1.")
        if num_nodes <= 1 and fraction_major != 1.0: # If only 1 node, it gets all data
            if num_nodes == 1: return [self.df.copy().reset_index(drop=True)]
            raise ValueError("Non-uniform partition needs at least 2 nodes unless fraction_major is 1.")

        partitions = []
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True) # Ensure shuffling
        
        major_size = int(len(df_shuffled) * fraction_major)
        partitions.append(df_shuffled.iloc[:major_size].reset_index(drop=True))
        
        remaining_df = df_shuffled.iloc[major_size:]
        if num_nodes > 1:
            if len(remaining_df) > 0:
                minor_nodes_count = num_nodes - 1
                # np.array_split handles empty arrays correctly if remaining_df is empty
                minor_partitions_indices = np.array_split(np.arange(len(remaining_df)), minor_nodes_count)
                for idx_group in minor_partitions_indices:
                    partitions.append(remaining_df.iloc[idx_group].reset_index(drop=True))
            else: # No remaining data for minor nodes
                for _ in range(num_nodes - 1):
                    partitions.append(pd.DataFrame(columns=self.df.columns))
        return partitions
    
    def highly_imbalanced_partition(self, num_nodes, fraction_major=0.9):
        """Partitions data with one node having a very large fraction (e.g., 90%)."""
        if not (0 < fraction_major < 1):
            raise ValueError("Fraction for major node must be between 0 and 1.")
        if num_nodes <= 1 and fraction_major != 1.0:
            if num_nodes == 1: return [self.df.copy().reset_index(drop=True)]
            raise ValueError("Highly imbalanced partition needs at least 2 nodes unless fraction_major is 1.")

        partitions = []
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        major_size = int(len(df_shuffled) * fraction_major)
        partitions.append(df_shuffled.iloc[:major_size].reset_index(drop=True))
        
        remaining_df = df_shuffled.iloc[major_size:]
        if num_nodes > 1:
            if len(remaining_df) > 0:
                minor_nodes_count = num_nodes - 1
                minor_partitions_indices = np.array_split(np.arange(len(remaining_df)), minor_nodes_count)
                for idx_group in minor_partitions_indices:
                    partitions.append(remaining_df.iloc[idx_group].reset_index(drop=True))
            else: # No remaining data for minor nodes
                for _ in range(num_nodes - 1):
                    partitions.append(pd.DataFrame(columns=self.df.columns))
        return partitions

def assign_data_to_nodes(partitions):
    """Assigns list of DataFrame partitions to a dictionary of nodes."""
    nodes = {}
    for i, partition in enumerate(partitions):
        nodes[f"node_{i+1}"] = partition
    return nodes 