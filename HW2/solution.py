import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def generate_signal(seed, total_samples=10000, duration=10):
    """
    Generates the noisy signals and pure targets.
    Returns a list of noisy signals (one per frequency) and pure targets.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    frequencies = [1, 3, 5, 7]

    noisy_sinusoids = []
    pure_targets = []

    for f in frequencies:
        # Noise changes at every sample
        # We use additive noise and amplitude modulation, but preserve phase coherence
        # to make the signal learnable.

        # Amplitude noise (Multiplicative)
        A = rng.uniform(0.5, 1.5, size=total_samples)

        # Additive Noise
        noise = rng.normal(0, 0.2, size=total_samples)

        # Pure target
        pure = np.sin(2 * np.pi * f * t)
        pure_targets.append(pure)

        # Noisy component: Amplitude modulated sine + additive noise
        # We removed the random phase per sample as it destroys the signal structure.
        noisy = A * pure + noise
        noisy_sinusoids.append(noisy)

    # Summation and Normalization
    # The assignment requires S to be a mix of all 4 frequencies.
    S_mixed = np.sum(noisy_sinusoids, axis=0) / 4.0

    return t, S_mixed, pure_targets


def create_dataset(S, targets, window_size=50, batch_size=32):
    """
    Creates a windowed dataset for training/testing.
    We create 4 separate datasets (one for each frequency selection) and combine them.
    """
    all_inputs = []
    all_targets = []

    total_samples = len(S)

    for i in range(4):
        # Create One-Hot vector for this frequency
        # Shape: (total_samples, 4)
        C = np.zeros((total_samples, 4))
        C[:, i] = 1.0

        # S is the mixed signal for all samples
        S_reshaped = S.reshape(-1, 1)
        features = np.hstack([S_reshaped, C])

        # Target for this frequency
        target = targets[i]

        # Create windows
        X_windows = []
        y_windows = []

        for j in range(total_samples - window_size):
            X_windows.append(features[j : j + window_size])
            y_windows.append(target[j + window_size])  # Predict the next point

        all_inputs.append(np.array(X_windows))
        all_targets.append(np.array(y_windows))

    # Concatenate all frequencies
    X_final = np.concatenate(all_inputs, axis=0)
    y_final = np.concatenate(all_targets, axis=0)

    return X_final, y_final


def main():
    # Parameters
    # Increased window size to capture more temporal context (100 samples = 0.1s)
    # This helps in distinguishing lower frequencies (like 1Hz)
    WINDOW_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 15

    print("Generating Data...")
    # Train Data (Seed 1)
    t_train, S_train, targets_train = generate_signal(seed=1)
    X_full, y_full = create_dataset(S_train, targets_train, window_size=WINDOW_SIZE)

    # Use sklearn to split into Train and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # Test Data (Seed 2) - Keep as separate hold-out
    t_test, S_test, targets_test = generate_signal(seed=2)
    X_test, y_test = create_dataset(S_test, targets_test, window_size=WINDOW_SIZE)

    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Build Model
    # Improved Architecture: CRNN (Conv1D + LSTM)
    # 1. Conv1D: Acts as a learnable filter to smooth high-frequency noise immediately.
    # 2. LSTM: Captures the long-term temporal dependency (the 3Hz rhythm).

    print("Building Model...")
    model = keras.Sequential(
        [
            keras.Input(shape=(WINDOW_SIZE, 5)),
            # Convolutional Filter
            layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
            layers.MaxPooling1D(
                pool_size=2
            ),  # Downsample to focus on dominant features
            # Recurrent Layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Train
    print("Training...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Evaluation
    print("Evaluating...")
    mse_test = model.evaluate(X_test, y_test, verbose="0")
    print(f"Test MSE: {mse_test:.6f}")

    # Visualization
    # We will visualize the prediction for Frequency 2 (3 Hz) on the Test set
    # We need to extract the subset of X_test corresponding to freq index 1

    # The data was concatenated in order 0, 1, 2, 3.
    # Each block has (10000 - 50) samples.
    samples_per_freq = 10000 - WINDOW_SIZE
    start_idx = 1 * samples_per_freq
    end_idx = 2 * samples_per_freq

    X_viz = X_test[start_idx:end_idx]
    y_true_viz = y_test[start_idx:end_idx]

    # Predict
    y_pred_viz = model.predict(X_viz)

    # Plot
    plt.figure(figsize=(12, 6))
    # Plot a subset of time to make it visible
    plot_len = 200

    # Reconstruct time axis for the windowed data
    # The target corresponds to t[window_size:]
    t_viz = t_test[WINDOW_SIZE:][:plot_len]

    # Get the noisy signal S for this segment
    # X_viz shape is (N, 50, 5). The last point in window is the current time t.
    # Feature 0 is S.
    S_viz = X_viz[:plot_len, -1, 0]

    plt.plot(t_viz, S_viz, label="Mixed Input (S)", alpha=0.3, color="gray")
    plt.plot(
        t_viz,
        y_true_viz[:plot_len],
        label="Target Pure (3Hz)",
        linewidth=2,
        color="green",
    )
    plt.plot(
        t_viz, y_pred_viz[:plot_len], label="LSTM Output", linestyle="--", color="red"
    )

    plt.title(
        f"Frequency Extraction Results (f=3Hz) - Test Set\nMSE: {mse_test:.5f}\n(Note: Input S is mixed, so it does not visually track the single target)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    output_path = "/workspaces/llms-and-multi-agent-orchestration/HW2/result_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
