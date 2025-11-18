# M.Sc. Assignment: Developing an LSTM System for Frequency Extraction from a Mixed Signal

**Dr. Segal Yoram**
**All rights reserved - Dr. Segal Yoram**
**November 2025**

---

## Table of Contents

* 1 Background and Goal
  * 1.1 Problem Statement
  * 1.2 Principle
  * 1.3 Usage Example
* 2 Dataset Creation
  * 2.1 General Parameters
  * 2.2 Noisy Signal Creation (Mixed and Noisy Signal S)
  * 2.3 Ground Truth Targets Creation (Targets)
  * 2.4 Train vs. Test Sets
* 3 Training Dataset Structure
* 4 Pedagogical Emphasis: Internal State Management - Sequence Length=1
  * 4.1 The Internal State of LSTM
  * 4.2 Critical Implementation Requirements ($L=1$)
  * 4.3 Alternative and Justification: Recommendation for Improvement and Justification Requirement
* 5 Performance Evaluation
  * 5.1 Success Metrics
  * 5.2 Recommended Graphs
* 6 Assignment Summary
* 7 References in Hebrew

---

## 1 Background and Goal

### 1.1 Problem Statement

Given a mixed and noisy signal S composed of 4 different sinusoidal frequencies. The noise changes randomly at every sample.

The goal is to develop an LSTM (Long Short-Term Memory) network capable of extracting each pure frequency separately from the mixed signal, while completely ignoring the noise.

### 1.2 The Principle

The system is required to perform Conditional Regression:

**Table 1: System Input/Output Structure**

| Input | Description | Target Output |
| :--- | :--- | :--- |
| S[t] | Sample from the noisy signal | Target[t] |
| C | One-Hot selection vector for frequency selection | |

### 1.3 Usage Example

If the selection vector is $C=[0 \ 1 \ 0 \ 0]$, we want to extract the pure frequency $f_{2}$:

Input: $\binom{S[t]}{C}$ Output: $Sinus_{pure}[t]$

* $S[0]+C \rightarrow LSTM \rightarrow Sinus_{2}^{pure}[0]$ (pure)
* $S[1]+C \rightarrow LSTM \rightarrow Sinus_{2}^{pure}[1]$ (pure)

---

## 2 Dataset Creation

### 2.1 General Parameters

* **Frequencies:** $f_{1}=1$ Hz, $f_{2}=3$ Hz, $f_{3}=5$ Hz, $f_{4}=7$ Hz
* **Time range:** 0 - 10 seconds
* **Sampling Rate:** (Not specified in source, but total samples implies 1000 Hz)
* **Total Samples:** 10,000

### 2.2 Noisy Signal Creation (Mixed and Noisy Signal S)

Critical point: The noise (amplitude $A_{i}(t)$ and phase $\phi_{i}(t)$) must change at every sample t.

1. **The noisy sinus at sample t:**
    * Amplitude: $A_{i}(t)\sim Uniform(0.8,1.2)$
    * Phase: $\phi_{i}(t)\sim Uniform(0,2\pi)$
    * $$sinus_{i}^{noisy}(t)=A_{i}(t)\cdot sin(2\pi\cdot f_{i}\cdot t+\phi_{i}(t))$$

2. **Summation and Normalization (System Input):**
    * $$S(t)=\frac{1}{4}\sum_{i=1}^{4}Sinus_{i}^{noisy}(t)$$

### 2.3 Ground Truth Targets Creation

The pure target for each frequency is:
$$Target_{i}(t)=sin(2\pi\cdot f_{i}\cdot t)$$

### 2.4 Train vs. Test Sets Differences

* **Training set:** Uses a random set (Seed #1).
* **Test set:** Uses a random set (Seed #2). (Same frequencies, completely different noise!)

---

## 3 Training Dataset Structure

Total rows in training set 40,000 (10,000 samples * 4 frequencies). Data format: each row represents a single sample. The network input is a vector of size 5: `(S[t] C1 C2 C3 C4)`

**Table 2: Example Data Format (Training Set)**

| # | Time | S (Noisy Input) | C (Selection) | Target (Pure Output) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.000 | 0.8124 | 1,0,0,0 | 0.0000 |
| ... | | | | |
| 10001 | 0.000 | 0.8124 | 0,1,0,0 | 0.0000 |
| 10002 | 0.001 | 0.7932 | 0,1,0,0 | 0.0188 |
| ... | | | | |
| 40000 | 9.999 | 0.6543 | 0,0,0,1 | 0.0440 |

---

## 4 Pedagogical Emphasis: Internal State Management

### Sequence Length=1

Within this assignment, we define the Sequence Length $L=1$ as a methodological default.

### 4.1 The Internal State of LSTM

The internal state of an LSTM is composed of the Hidden State ($h_{t}$) and the Cell State ($c_{t}$). This state allows the network to learn Temporal Dependency between samples.

### 4.2 Critical Implementation Requirements ($L=1$)

When working with $L=1$, we must manually manage the internal state in the training loop so the network can utilize its memory:

* You must ensure the internal state ($h_{t}, c_{t}$) is not Reset between consecutive samples.

**Table 3: Comparison: State Management in an LSTM model**

| Scenario | Required Action | Explanation | Significance |
| :--- | :--- | :--- | :--- |
| $L>1$ (Regular) | Reset state with each sequence feed. | The network assumes no serial connection between sequences. | |
| $L=1$ (This Assignment) | Preserve state and pass it as input to the next step. | The network *can* learn serial patterns via state management. | **Critical** |

### 4.3 Alternative and Justification: Recommendation for Improvement and Justification Requirement

Disclaimer: Training with long sequences ($L>1$) is more computationally efficient and pedagogically effective for demonstrating the full advantage of LSTM.

Students are invited to work with $L \ne 1$ (Sliding Window) of size $L=10$ or $L=50$ instead of $L=1$.

**Justification Requirement:** If $L \ne 1$ is chosen, a detailed justification must be included in the assignment presenting the choice, how it contributes to the Temporal advantage of the LSTM, and the method of handling the output.

---

## 5 Performance Evaluation

### 5.1 Success Metrics

1. **MSE on Training Set (Noise #1):**
    $$MSE_{train}=\frac{1}{40000}\sum_{j=1}^{40000}(LSTM(S_{train}[t],C)-Target[t])^{2}$$

2. **MSE on Test Set (Noise #2):**
    $$MSE_{test}=\frac{1}{40000}\sum_{j=1}^{40000}(LSTM(S_{test}[t],C)-Target[t])^{2}$$

3. **Generalization Check:** If $MSE_{test}\approx MSE_{train}$, then the system generalizes well!

### 5.2 Recommended Graphs

A visual comparison on the test set (Noise #2) should be presented, such as:

1. **Graph 1: Comparison for a single frequency ($f_{2}$):** Display three components on the same graph: 1. Target (line), 2. LSTM Output (dots), 3. Mixed and Noisy S (chaotic background
