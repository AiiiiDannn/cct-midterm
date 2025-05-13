# Cultural Consensus Theory (CCT) Midterm Report

## 1. Objective and Background

This project implements a basic **Cultural Consensus Theory (CCT)** model using PyMC to analyze a small, simulated dataset about local plant knowledge. The goal is to estimate:

- The **consensus answer** ( \( Z_j \) ) for each question    
- The **knowledge competence** ( \( D_j \) ) of each informant

The core idea behind CCT is that the more someone knows about the cultural consensus, the more their answers will agree with others — and especially with other knowledgeable informants.

---

## 2. Assignment Requirements and Implementation

### Data Loading

- The function `load_plant_data()` loads data from `plant_knowledge.csv`  
- The first column ("Informant ID") is dropped  
- A NumPy matrix \( X_{ij} \in \{0,1\}^{N \times M} \) is returned  
- The file path is resolved relative to the script location for portability

### Model Definition

- Definitions:  
  - \( D_i \): competence of informant \( i \), probability of giving correct answer  
  - \( Z_j \): consensus answer (latent ground truth) for item \( j \)  
  - \( X_{ij} \): binary response by informant \( i \) to item \( j \)  
- The response probability is defined as:  
  \[  
  p_{ij} = Z_j \cdot D_i + (1 - Z_j)(1 - D_i)  
  \]  
- Model components:  
  - \( D_i \sim \text{Uniform}(0.5, 1) \)    
  - \( Z_j \sim \text{Bernoulli}(0.5) \)    
  - \( X_{ij} \sim \text{Bernoulli}(p_{ij}) \)

### Justification for Priors

- **Competence \( D_i \)**:  
  - A uniform prior from 0.5 to 1 reflects minimal assumptions:  
    - 0.5 is the expected accuracy from random guessing  
    - 1.0 represents perfect knowledge  
  - This range ensures that informants are at least somewhat knowledgeable

- **Consensus answer \( Z_j \)**:  
  - Bernoulli(0.5) prior assumes complete uncertainty about the truth  
  - It equally allows the answer to be either 0 or 1 before seeing data

---

## 3. Inference Setup

- Inference is performed using PyMC's `pm.sample()`:  
  - `draws=2000`, `chains=4`, `tune=1000`, `target_accept=0.95`  
- NUTS sampler is used for continuous \( D \), and BinaryGibbsMetropolis for discrete \( Z \)  
- All chains converged (`R_hat = 1.0`)  
- No divergent transitions  
- Effective sample sizes (ESS) are large, indicating robust sampling

---

## 4. Results and Analysis

### Did the model converge?

Yes. All R-hat values were exactly 1.00. Sampling diagnostics show no issues.

### Competence Estimation (Di)

Posterior mean competence per informant:

```text
Informant 1: 0.572
Informant 2: 0.764
Informant 3: 0.560 ← least competent
Informant 4: 0.569
Informant 5: 0.797
Informant 6: 0.877 ← most competent
Informant 7: 0.685
Informant 8: 0.694
Informant 9: 0.563
Informant 10: 0.567
```

**Conclusion**:

* All informants are above guessing level (0.5), showing real knowledge
* Informant 6 is the most competent
* Informant 3 has the lowest competence

---

### Consensus Answer Estimation (Zj)

Posterior mean probabilities for each question:

```text
[0.011, 0.841, 0.990, 0.015, 0.546, 0.938, 0.998, 0.923, 0.977, 0.928,
 0.985, 0.009, 0.258, 0.912, 0.253, 0.020, 0.008, 0.059, 0.032, 0.997]
```

Consensus answer key (thresholded at 0.5):

```text
[0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1]
```

**Conclusion**:

* Most questions have clear consensus (posterior near 0 or 1)
* A few questions (like #5) are borderline, indicating group disagreement
* The model still resolves them using informant competence weighting

---

### Comparison with Majority Vote

Majority vote answers:

```text
[0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1]
```

Model-based consensus:

```text
[0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1]
```

**Agreement rate = 0.75**

**Conclusion**:

* The model disagrees with the majority vote on 5 out of 20 questions
* This happens when:
  * High-competence informants give a minority answer
  * The majority is incorrect due to low-competence informants
* The model gives more weight to informed individuals, producing better estimates

---


## 5. Project Structure

```text
cct-midterm/
├── code/
│   └── cct.py               # Main model and analysis script
├── data/
│   └── plant_knowledge.csv  # Raw data
├── README.md                # This report
```

---

## 6. Final Notes

* All code runs in the class container
* Functions are modular and clearly documented
* Inference results match expectations and outputs
* When modeling decisions were uncertain, I followed Bayesian best practices and documented them
* ChatGPT (model: Cognitive Model & ChatGPT o4-mini-high) was used as a reference throughout the development process, assisting in clarifying assignment requirements, shaping the initial code structure and generation, and supporting code debugging.