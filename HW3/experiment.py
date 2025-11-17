import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer


# Compute cosine similarity for each pair
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get the base directory
base_dir = Path(__file__).parent.resolve()

# Create assets directory if it doesn't exist
assets_dir = base_dir / "assets"
assets_dir.mkdir(exist_ok=True)

# Open output file for writing
output_file = assets_dir / "experiment_results.txt"

# Process each error percentage
error_percentages = ["0_percent_errors", "25_percent_errors", "50_percent_errors"]
error_pct_values = [0, 25, 50]  # For graphing
all_avg_similarities = []  # Store average similarities for graphing
all_similarities_by_error = []  # Store all similarities for each error percentage

with open(output_file, "w") as out:
    for error_pct in error_percentages:
        out.write(f"\n{'=' * 80}\n")
        out.write(f"Processing: {error_pct}\n")
        out.write(f"{'=' * 80}\n\n")

        # Load original and translated sentences from JSON files
        with open(base_dir / "data" / f"{error_pct}.json", "r") as f:
            original_sentences = json.load(f)

        with open(base_dir / "output" / f"{error_pct}_results.json", "r") as f:
            translated_sentences = json.load(f)

        out.write(f"Found {len(original_sentences)} original sentences\n")
        out.write(f"Found {len(translated_sentences)} translated sentences\n\n")

        # Compute embeddings
        original_embeddings = model.encode(original_sentences)
        translated_embeddings = model.encode(translated_sentences)

        # Compute cosine similarity for each pair
        similarities = []
        out.write("Cosine Similarities between sentence pairs:\n")
        out.write("-" * 80 + "\n")
        for i in range(min(len(original_sentences), len(translated_sentences))):
            similarity = cosine_similarity(
                original_embeddings[i], translated_embeddings[i]
            )
            similarities.append(similarity)
            out.write(f"\nPair {i + 1}:\n")
            out.write(f"Original:    {original_sentences[i][:80]}...\n")
            out.write(f"Translated:  {translated_sentences[i][:80]}...\n")
            out.write(f"Similarity:  {similarity:.4f}\n")
            out.write("-" * 80 + "\n")

        # Store for graphing
        all_avg_similarities.append(np.mean(similarities))
        all_similarities_by_error.append(similarities)

        # Write summary statistics
        out.write(f"\nSummary for {error_pct}:\n")
        out.write(f"Average Similarity: {np.mean(similarities):.4f}\n")
        out.write(f"Min Similarity:     {np.min(similarities):.4f}\n")
        out.write(f"Max Similarity:     {np.max(similarities):.4f}\n")
        out.write(f"Std Deviation:      {np.std(similarities):.4f}\n")

# Create the graph
plt.figure(figsize=(12, 6))

# Plot individual sentence similarities
for i in range(len(all_similarities_by_error[0])):  # Number of sentences
    sentence_similarities = [
        all_similarities_by_error[j][i] for j in range(len(error_pct_values))
    ]
    plt.plot(
        error_pct_values,
        sentence_similarities,
        "o-",
        alpha=0.3,
        linewidth=1,
        label=f"Sentence {i + 1}",
    )

# Plot average similarity with emphasis
plt.plot(
    error_pct_values,
    all_avg_similarities,
    "ro-",
    linewidth=3,
    markersize=10,
    label="Average",
    zorder=10,
)

plt.xlabel("Spelling Error Percentage (%)", fontsize=12)
plt.ylabel("Cosine Similarity", fontsize=12)
plt.title(
    "Impact of Spelling Errors on Round-Trip Translation Quality",
    fontsize=14,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.xticks(error_pct_values)
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the graph
graph_file = assets_dir / "cosine_similarity_graph.png"
plt.savefig(graph_file, dpi=300, bbox_inches="tight")

print(f"Results written to: {output_file}")
print(f"Graph saved to: {graph_file}")
