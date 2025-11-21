import matplotlib.pyplot as plt
import numpy as np


def generate_clean_mix(duration=2, total_samples=2000, noise_level=0.2):
    t = np.linspace(0, duration, total_samples, endpoint=False)
    frequencies = [1, 3, 5, 7]

    pure_sinusoids = []

    # Generate pure sines
    for f in frequencies:
        pure = np.sin(2 * np.pi * f * t)
        pure_sinusoids.append(pure)

    # Mix them (Sum / 4)
    S_clean_mixed = np.sum(pure_sinusoids, axis=0) / 4.0

    # Add light noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, noise_level, size=total_samples)
    S_noisy_mixed = S_clean_mixed + noise

    return t, S_noisy_mixed, pure_sinusoids


def main():
    t, S_mixed, components = generate_clean_mix()

    # We want to visualize the 3Hz component (index 1)
    target_3hz = components[1]

    plt.figure(figsize=(12, 8))

    # Plot 1: The Clean Mixed Signal vs The Target
    plt.subplot(2, 1, 1)
    plt.plot(
        t,
        S_mixed,
        label="Mixed Signal + Light Noise",
        color="gray",
        linewidth=1.0,
        alpha=0.8,
    )
    plt.plot(
        t,
        target_3hz,
        label="Target Component (3Hz)",
        color="green",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Mixed Signal with Light Noise vs Target")
    plt.legend()
    plt.grid(True)

    # Plot 2: Decomposition
    plt.subplot(2, 1, 2)
    plt.plot(t, components[0], label="1 Hz", alpha=0.5)
    plt.plot(t, components[1], label="3 Hz (Target)", linewidth=2, color="green")
    plt.plot(t, components[2], label="5 Hz", alpha=0.5)
    plt.plot(t, components[3], label="7 Hz", alpha=0.5)
    plt.title("Individual Components")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    output_path = (
        "/workspaces/llms-and-multi-agent-orchestration/HW2/clean_mix_demo.png"
    )
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
