import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')


# Mount Google Drive
drive.mount('/content/drive')


def load_and_preprocess_data(file_path):
    """
    Load EEG data from CSV and preprocess it
    """
    # Read CSV file
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
   
    # EEG channels
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3',
                'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
   
    print(f"Data shape: {data.shape}")
    return data[channels]


def bandpass_filter(data, fs=250, order=4):
    """
    Apply bandpass filters to extract different frequency bands
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
   
    filtered_data = {}
   
    for band_name, (low, high) in bands.items():
        print(f"Filtering {band_name} band ({low}-{high} Hz)")
        # Create bandpass filter
        nyquist = fs / 2
        low_normalized = low / nyquist
        high_normalized = high / nyquist
        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
       
        # Apply filter to each channel
        filtered_data[band_name] = pd.DataFrame()
        for column in data.columns:
            filtered_data[band_name][column] = signal.filtfilt(b, a, data[column])
   
    return filtered_data


def compute_fuzzy_entropy(data, m=2, r=0.2, n=None):
    """
    Compute Fuzzy Entropy for a time series
    """
    N = len(data)
    if n is None:
        n = N - m
   
    # Normalize the data
    data = (data - np.mean(data)) / np.std(data)
    r = r * np.std(data)
   
    # Initialize similarity counts
    phi = np.zeros(2)
   
    for k in range(2):
        m_k = m + k
        patterns = np.zeros((n, m_k))
       
        # Create patterns of length m and m+1
        for i in range(n):
            patterns[i] = data[i:i + m_k]
       
        # Calculate distances between patterns
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i,j] = np.max(np.abs(patterns[i] - patterns[j]))
       
        # Calculate fuzzy membership
        D = np.exp(-np.power(dist, 2) / r)
       
        # Sum similarities
        phi[k] = np.sum(D) / (n * (n-1))
   
    return -np.log(phi[1] / phi[0])


def compute_mfe(data, scale_factors=range(1, 21), m=2, r=0.2):
    """
    Compute Multiscale Fuzzy Entropy
    """
    mfe = np.zeros(len(scale_factors))
   
    for idx, scale in enumerate(scale_factors):
        # Coarse-grain the time series
        coarse_grained = np.array([np.mean(data[i:i+scale])
                                 for i in range(0, len(data)-scale+1, scale)])
       
        # Compute fuzzy entropy for this scale
        mfe[idx] = compute_fuzzy_entropy(coarse_grained, m=m, r=r)
   
    return mfe


def create_epochs(data, epoch_length=750):
    """
    Create epochs from data with proper error handling
    """
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    print(f"Epoch length: {epoch_length}")
   
    # Calculate number of complete epochs
    n_epochs = total_samples // epoch_length
    print(f"Number of complete epochs: {n_epochs}")
   
    if n_epochs == 0:
        # If data is shorter than epoch_length, use the entire data as one epoch
        print("Warning: Data length is shorter than epoch length. Using entire data as one epoch.")
        return [data]
   
    # Calculate how many samples to use (truncate incomplete epochs)
    samples_to_use = n_epochs * epoch_length
   
    # Split into epochs
    epochs = np.array_split(data[:samples_to_use], n_epochs)
    print(f"Created {len(epochs)} epochs")
   
    return epochs


def process_eeg_data(data, epoch_length=750):  # 3 seconds at 250 Hz
    """
    Process EEG data with epoching and MFE calculation
    """
    results = {}
   
    # Normalize data
    print("Normalizing data...")
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
   
    # Apply bandpass filtering
    print("Applying bandpass filters...")
    filtered_data = bandpass_filter(normalized_data)
   
    # Calculate MFE for each frequency band and channel
    for band_name, band_data in filtered_data.items():
        print(f"\nProcessing {band_name} band...")
        results[band_name] = {}
       
        for channel in band_data.columns:
            print(f"Processing channel {channel}")
           
            # Create epochs
            epochs = create_epochs(band_data[channel].values, epoch_length)
           
            # Calculate MFE for each epoch
            mfe_values = []
            for i, epoch in enumerate(epochs):
                mfe = compute_mfe(epoch)
                mfe_values.append(mfe)
           
            # Store mean MFE across epochs
            results[band_name][channel] = np.mean(mfe_values, axis=0)
   
    return results


def visualize_results(results, output_path):
    """
    Create visualizations of the results
    """
    print("Creating visualizations...")
    # Plot MFE values for each frequency band
    for band_name, band_results in results.items():
        plt.figure(figsize=(12, 8))
       
        for channel, mfe_values in band_results.items():
            plt.plot(range(1, len(mfe_values) + 1), mfe_values, label=channel)
       
        plt.title(f'Multiscale Fuzzy Entropy - {band_name} band')
        plt.xlabel('Scale Factor')
        plt.ylabel('Fuzzy Entropy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_path}/mfe_{band_name}.png')
        plt.close()
def analyze_and_visualize_results(results, output_path, groups=None):
    """
    Comprehensive analysis and visualization of MFE results
    Parameters:
        results: Dictionary containing MFE values for each band and channel
        output_path: Path to save visualizations
        groups: Dictionary mapping subject IDs to groups (AD or HC) if available
    """
    print("\nGenerating comprehensive analysis...")
   
    # 1. Basic MFE Visualization
    plot_mfe_distributions(results, output_path)
   
    # 2. Statistical Analysis
    statistical_analysis(results, output_path)
   
    # 3. Channel Comparison
    plot_channel_comparison(results, output_path)
   
    # 4. Band Power Analysis
    plot_band_power_analysis(results, output_path)
   
    if groups is not None:
        # 5. Group Comparison (if group information is available)
        plot_group_comparison(results, groups, output_path)


def plot_mfe_distributions(results, output_path):
    """Plot MFE distributions for each frequency band"""
    for band_name, band_results in results.items():
        plt.figure(figsize=(15, 10))
       
        # Create violin plots for each channel
        data_for_plot = []
        labels = []
        for channel, mfe_values in band_results.items():
            data_for_plot.append(mfe_values)
            labels.append(channel)
       
        sns.violinplot(data=data_for_plot)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.title(f'MFE Distribution - {band_name} Band')
        plt.ylabel('Fuzzy Entropy')
        plt.tight_layout()
        plt.savefig(f'{output_path}/mfe_distribution_{band_name}.png')
        plt.close()


def statistical_analysis(results, output_path):
    """Perform statistical analysis on MFE values"""
    stats_results = {}
   
    for band_name, band_results in results.items():
        stats_results[band_name] = {
            'mean': {},
            'std': {},
            'median': {},
            'iqr': {}
        }
       
        for channel, mfe_values in band_results.items():
            stats_results[band_name]['mean'][channel] = np.mean(mfe_values)
            stats_results[band_name]['std'][channel] = np.std(mfe_values)
            stats_results[band_name]['median'][channel] = np.median(mfe_values)
            stats_results[band_name]['iqr'][channel] = np.percentile(mfe_values, 75) - np.percentile(mfe_values, 25)
   
    # Save statistical results
    with open(f'{output_path}/statistical_analysis.txt', 'w') as f:
        for band, stats in stats_results.items():
            f.write(f"\n{band} Band Statistics:\n")
            f.write("=" * 50 + "\n")
            for metric, values in stats.items():
                f.write(f"\n{metric.upper()}:\n")
                for channel, value in values.items():
                    f.write(f"{channel}: {value:.4f}\n")


def plot_channel_comparison(results, output_path):
    """Create heatmap of channel relationships"""
    for band_name, band_results in results.items():
        # Create correlation matrix
        channels = list(band_results.keys())
        corr_matrix = np.zeros((len(channels), len(channels)))
       
        for i, ch1 in enumerate(channels):
            for j, ch2 in enumerate(channels):
                corr_matrix[i,j] = np.corrcoef(band_results[ch1], band_results[ch2])[0,1]
       
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, xticklabels=channels, yticklabels=channels,
                   cmap='coolwarm', center=0, annot=True, fmt='.2f')
        plt.title(f'Channel Correlation - {band_name} Band')
        plt.tight_layout()
        plt.savefig(f'{output_path}/channel_correlation_{band_name}.png')
        plt.close()


def plot_band_power_analysis(results, output_path):
    """Plot average MFE across different frequency bands"""
    avg_band_mfe = {}
    for band_name, band_results in results.items():
        avg_band_mfe[band_name] = np.mean([np.mean(mfe) for mfe in band_results.values()])
   
    plt.figure(figsize=(10, 6))
    plt.bar(avg_band_mfe.keys(), avg_band_mfe.values())
    plt.title('Average MFE Across Frequency Bands')
    plt.ylabel('Average Fuzzy Entropy')
    plt.tight_layout()
    plt.savefig(f'{output_path}/average_band_mfe.png')
    plt.close()


def compute_summary_statistics(results):
    """
    Compute summary statistics for each frequency band and channel
    """
    summary_stats = {}
   
    for band_name, band_results in results.items():
        summary_stats[band_name] = {
            'mean_mfe': np.mean([np.mean(mfe) for mfe in band_results.values()]),
            'std_mfe': np.std([np.mean(mfe) for mfe in band_results.values()]),
            'channel_stats': {}
        }
       
        for channel, mfe_values in band_results.items():
            summary_stats[band_name]['channel_stats'][channel] = {
                'mean': np.mean(mfe_values),
                'std': np.std(mfe_values),
                'median': np.median(mfe_values),
                'max': np.max(mfe_values),
                'min': np.min(mfe_values)
            }
   
    return summary_stats


def plot_band_comparison(results, output_path):
    """
    Create box plots comparing MFE values across frequency bands
    """
    plt.figure(figsize=(15, 8))
   
    band_data = []
    band_labels = []
   
    for band_name, band_results in results.items():
        for channel, mfe_values in band_results.items():
            band_data.extend(mfe_values)
            band_labels.extend([band_name] * len(mfe_values))
   
    sns.boxplot(x=band_labels, y=band_data)
    plt.title('MFE Distribution Across Frequency Bands')
    plt.xlabel('Frequency Band')
    plt.ylabel('Fuzzy Entropy')
    plt.savefig(f'{output_path}/band_comparison_boxplot.png')
    plt.close()


def create_topographic_map(results, output_path):
    """
    Create topographic maps of MFE values
    """
    for band_name, band_results in results.items():
        plt.figure(figsize=(12, 8))
       
        # Calculate mean MFE for each channel
        channel_means = {ch: np.mean(mfe) for ch, mfe in band_results.items()}
       
        # Create heatmap
        channel_matrix = np.zeros((4, 5))  # Approximate 10-20 system layout
        channel_positions = {
            'Fp1': (0, 1), 'Fp2': (0, 3),
            'F7': (1, 0), 'F3': (1, 1), 'Fz': (1, 2), 'F4': (1, 3), 'F8': (1, 4),
            'T3': (2, 0), 'C3': (2, 1), 'Cz': (2, 2), 'C4': (2, 3), 'T4': (2, 4),
            'T5': (3, 0), 'P3': (3, 1), 'Pz': (3, 2), 'P4': (3, 3), 'T6': (3, 4)
        }
       
        for ch, pos in channel_positions.items():
            if ch in channel_means:
                channel_matrix[pos] = channel_means[ch]
       
        sns.heatmap(channel_matrix, cmap='coolwarm', center=np.mean(list(channel_means.values())))
        plt.title(f'Topographic MFE Distribution - {band_name} Band')
        plt.savefig(f'{output_path}/topographic_map_{band_name}.png')
        plt.close()


def perform_statistical_tests(results):
    """
    Perform statistical tests between frequency bands
    """
    stats_results = {}
    bands = list(results.keys())
   
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            band1_data = np.concatenate([mfe for mfe in results[bands[i]].values()])
            band2_data = np.concatenate([mfe for mfe in results[bands[j]].values()])
           
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(band1_data, band2_data)
            stats_results[f'{bands[i]} vs {bands[j]}'] = {
                'statistic': statistic,
                'p_value': p_value
            }
   
    return stats_results


def generate_final_report(results, output_path):
    """
    Generate a comprehensive final report
    """
    summary_stats = compute_summary_statistics(results)
    statistical_tests = perform_statistical_tests(results)
   
    with open(f'{output_path}/final_report.txt', 'w') as f:
        f.write("EEG Multiscale Fuzzy Entropy Analysis Report\n")
        f.write("==========================================\n\n")
       
        # Summary statistics
        f.write("1. Summary Statistics\n")
        f.write("-----------------\n")
        for band_name, stats in summary_stats.items():
            f.write(f"\n{band_name} Band:\n")
            f.write(f"Mean MFE: {stats['mean_mfe']:.4f} ± {stats['std_mfe']:.4f}\n")
           
            f.write("\nChannel-wise statistics:\n")
            for channel, ch_stats in stats['channel_stats'].items():
                f.write(f"{channel}:\n")
                for stat_name, value in ch_stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
       
        # Statistical tests
        f.write("\n2. Statistical Tests\n")
        f.write("------------------\n")
        for comparison, stats in statistical_tests.items():
            f.write(f"\n{comparison}:\n")
            f.write(f"U-statistic: {stats['statistic']:.4f}\n")
            f.write(f"p-value: {stats['p_value']:.4f}\n")
            f.write(f"Significant: {'Yes' if stats['p_value'] < 0.05 else 'No'}\n")
def main():
    # File paths
    input_path = '/content/drive/MyDrive/fuzzy/processed_s00.csv'  
    output_path = '/content/drive/MyDrive/fuzzy'     
    # Load and process data
    print("\nStarting EEG analysis...")
    print("========================")
   
    raw_data = load_and_preprocess_data(input_path)
   
    epoch_length = min(750, len(raw_data))
   
    print("\nProcessing EEG data...")
    print("=====================")
    results = process_eeg_data(raw_data, epoch_length=epoch_length)
   
    print("\nGenerating visualizations...")
    print("=========================")
    visualize_results(results, output_path)
   
    print("\nAnalysis complete!")
    print("\nGenerating comprehensive analysis...")
    plot_band_comparison(results, output_path)
    create_topographic_map(results, output_path)
    generate_final_report(results, output_path)
   
    print("\nAnalysis complete! Check the output folder for:")
    print("1. Band comparison plots")
    print("2. Topographic maps")
    print("3. Comprehensive final report")
    print("4. Statistical analysis results")


if __name__ == "__main__":
    main()




