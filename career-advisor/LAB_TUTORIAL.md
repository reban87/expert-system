# Lab Tutorial: Building & Understanding a Rule-Based Expert System
## Project: Career Path Advisor

**Course Module:** Artificial Intelligence / Knowledge-Based Systems  
**Estimated Lab Duration:** 2 to 3 hours  
**Prerequisites:** Basic Python, familiarity with dictionaries and dataclasses  

---

## 1. Learning Objectives

By the end of this lab, students will be able to:
1. **Define** the core components of an Expert System: *Knowledge Base*, *Working Memory*, *Inference Engine*, and *User Interface*.
2. **Differentiate** between *Forward Chaining* (data-driven reasoning) and *Backward Chaining* (goal-driven reasoning).
3. **Trace and calculate** confidence scores using forward-chaining matching logic.
4. **Extend** an existing production-style knowledge base with new domain rules.
5. **Deploy and test** the interactive system locally using modern Python tooling (`uv` and `Streamlit`).

---

## 2. Theoretical Foundations: What is an Expert System?

An **Expert System (ES)** is an AI system designed to emulate the decision-making ability of a human expert in a specific domain. Rather than learning patterns statistically from large datasets (as in modern Deep Learning), an expert system uses **explicit, human-readable rules and facts** provided by domain experts.

### 2.1 The Classic Architecture of an Expert System

```
                  ┌────────────────────────────────────────┐
                  │              HUMAN USER                │
                  └───────────────────▲────────────────────┘
                                      │  User Input / Advice
                                      ▼
                  ┌────────────────────────────────────────┐
                  │        USER INTERFACE (UI)             │
                  │             (app.py)                   │
                  └───────────────────▲────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
  ┌─────────────────────────┐                   ┌─────────────────────────┐
  │     WORKING MEMORY      │                   │   EXPLANATION FACILITY  │
  │ (User's Current Facts)  │                   │ (Scores, Trait Matches) │
  └────────────┬────────────┘                   └────────────▲────────────┘
               │                                             │
               ▼                                             │
  ┌──────────────────────────────────────────────────────────┴────────────┐
  │                           INFERENCE ENGINE                            │
  │                        (inference_engine.py)                          │
  │            - Evaluates rules against Working Memory facts             │
  │            - Executes Forward Chaining algorithm                      │
  │            - Computes confidence percentages                          │
  └───────────────────────────────────▲───────────────────────────────────┘
                                      │ Queries rules
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │                            KNOWLEDGE BASE                             │
  │                         (knowledge_base.py)                           │
  │              - Domain Ontology (TRAIT_OPTIONS)                        │
  │              - IF-THEN Rules (CAREER_RULES)                           │
  └───────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility in this Lab | Implementation File |
|---|---|---|
| **Knowledge Base (KB)** | Stores permanent domain knowledge as production rules (`IF traits THEN career`). | `knowledge_base.py` |
| **Working Memory** | Stores temporary session facts (the user's selected traits). | Streamlit session in `app.py` |
| **Inference Engine (IE)**| Applies reasoning rules to facts to derive conclusions and confidence scores. | `inference_engine.py` |
| **User Interface (UI)** | Gathers inputs from the user and displays ranked recommendations with explanations. | `app.py` |
| **Explanation Facility** | Shows why a career was recommended (matched traits count, roadmaps, skills). | `app.py` (expander & progress bars) |

---

### 2.2 Forward Chaining vs. Backward Chaining

- **Forward Chaining (Data-Driven):**  
  Starts with known facts and moves forward by checking rules whose conditions (antecedents) are satisfied, generating new conclusions (consequents).  
  *Example in this lab:* "Given that the user likes Technology, has Coding skills, and is Logical $\rightarrow$ infer they would succeed as a Software Engineer."
- **Backward Chaining (Goal-Driven):**  
  Starts with a hypothesis/goal and works backwards to see if available facts support it.  
  *Example:* "Is this user a fit for Software Engineer? Let's check if they have Coding skills."

> In this lab, we use **Forward Chaining** because we start with user profile facts and search through all career goals simultaneously.

---

## 3. Mathematical Model: Confidence Scoring

Each career rule $R_i$ contains $N$ required conditions:

$$\text{conditions}(R_i) = \{C_1, C_2, \dots, C_N\}$$

The user supplies their trait profile as a set of active facts $U$. For each rule, the inference engine counts how many conditions match:

$$M(R_i, U) = \sum_{j=1}^{N} \mathbf{1}\left(U[\text{category}_j] == C_j\right)$$

The **Confidence Percentage** is computed as:

$$\text{Confidence}(R_i) = \left( \frac{M(R_i, U)}{N} \right) \times 100\%$$

If $\text{Confidence}(R_i) \ge \text{Threshold}$ (default $40\%$), the recommendation is returned to the user, ranked in descending order by confidence.

---

## 4. Codebase Architecture & Walkthrough

### 4.1 Knowledge Base (`knowledge_base.py`)

The Knowledge Base separates the *domain ontology* from the *rules*:

1. **`TRAIT_OPTIONS`**: The vocabulary/ontology representing valid values across 5 dimensions:
   - *Primary Interest* (e.g., Technology & Computing, Healthcare & Medicine, Art & Creativity)
   - *Strongest Skill* (e.g., Analytical & Problem-Solving, Coding & Software, Empathy & People Skills)
   - *Core Value* (e.g., Innovation & Discovery, Helping Others, Financial Stability)
   - *Personality Type* (e.g., Logical & Analytical, Creative & Artistic, Empathetic & Caring)
   - *Education Background* (e.g., Computer Science / IT, Medicine / Nursing, Business / Management)

2. **`CareerRule` Dataclass**:
   ```python
   @dataclass
   class CareerRule:
       conditions: Dict[str, str]    # IF: {trait_category: expected_value}
       career: str                   # THEN: Recommended title
       field: str                    # Broader classification
       description: str              # Professional overview
       roadmap: str                  # Step-by-step career milestones
       skills_to_develop: List[str]  # Actionable skill prerequisites
       weight: float                 # Confidence weight
   ```

3. **`CAREER_RULES`**: A list of 12 production rules encoding careers from Software Engineer to Biomedical Researcher, Financial Analyst, and Teacher.

---

### 4.2 Inference Engine (`inference_engine.py`)

The `InferenceEngine` class acts on the facts:

```python
class InferenceEngine:
    def __init__(self, rules: List[CareerRule] = None, threshold: float = 40.0):
        self.rules = rules if rules is not None else CAREER_RULES
        self.threshold = threshold

    def evaluate(self, user_profile: Dict[str, str]) -> List[CareerRecommendation]:
        results: List[CareerRecommendation] = []

        for rule in self.rules:
            matched = 0
            total = len(rule.conditions)

            for trait_category, expected_value in rule.conditions.items():
                user_value = user_profile.get(trait_category)
                if user_value == expected_value:
                    matched += 1

            if total == 0:
                continue

            confidence = (matched / total) * 100.0

            if confidence >= self.threshold:
                results.append(CareerRecommendation(..., confidence=round(confidence, 1)))

        results.sort(key=lambda r: (r.confidence, r.matched_traits), reverse=True)
        return results
```

---

### 4.3 User Interface (`app.py`)

Built with **Streamlit**:
- **Sidebar:** Collects the 5 facts (traits) dynamically generated from `TRAIT_OPTIONS`.
- **Threshold Slider:** Allows the user to adjust the sensitivity threshold ($20\%$ to $80\%$).
- **Matching Panel:** Visualizes recommendations categorized into:
  - 🟢 **Excellent Match** ($\ge 80\%$)
  - 🔵 **Strong Match** ($60\% - 79\%$)
  - 🟡 **Partial Match** ($40\% - 59\%$)
- **Actionable Roadmaps:** Expandable containers showing concrete milestones and required skills.

---

## 5. Lab Setup Guide: Running Locally with `uv`

Instruct students to follow these steps in their terminal:

### Step 1: Clone or Navigate to the Workspace
```bash
cd /path/to/Expert-Systems
```

### Step 2: Initialize Virtual Environment using `uv`
```bash
# Recommended: Create environment targeting Python 3.12 or 3.11
uv venv --python 3.12

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
uv pip install -r career-advisor/requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run career-advisor/app.py
```
Open `http://localhost:8501` in your browser.

---

## 6. Student Hands-On Exercises & Assignments

### Exercise 1: Paper-and-Pencil Inference Trace (15 mins)
**Problem:** Suppose a student enters the following facts into the Working Memory:
- **Primary Interest:** `Technology & Computing`
- **Strongest Skill:** `Coding & Software`
- **Core Value:** `Innovation & Discovery`
- **Personality Type:** `Creative & Artistic`
- **Education Background:** `Business / Management`

**Task:**
1. Examine the `CAREER_RULES` for **Software Engineer** and **Data Scientist**.
2. Count the number of matched traits $M$ for each.
3. Calculate the exact Confidence percentage for both rules.
4. Verify your manual calculations by selecting these inputs in the running Streamlit web app.

---

### Exercise 2: Extending the Knowledge Base (30 mins)
**Objective:** Add a new career rule to `knowledge_base.py`.

**Task:** Add a rule for **"Cybersecurity Analyst"** with the following specification:
- **Conditions:**
  - *Primary Interest:* `Technology & Computing`
  - *Strongest Skill:* `Analytical & Problem-Solving`
  - *Core Value:* `Security & Stability`
  - *Personality Type:* `Detail-Oriented & Precise`
  - *Education Background:* `Computer Science / IT`
- **Field:** `Cybersecurity & IT Infrastructure`
- **Description:** Protects computer systems, networks, and sensitive data from cyber threats and unauthorized access.
- **Roadmap:**
  - Year 1: Master networking fundamentals (TCP/IP, OSI model) and Linux.
  - Year 2: Learn ethical hacking, SIEM tools, and obtain CompTIA Security+.
  - Year 3: Practice incident response, penetration testing, and security auditing.
- **Skills to Develop:** `Network Security`, `Ethical Hacking`, `SIEM Tools`, `Incident Response`, `Cryptography`
- **Weight:** `1.0`

*Test:* Reload the Streamlit app, choose these traits, and verify that "Cybersecurity Analyst" appears with a 100% match.

---

### Exercise 3: Adding Explanation of Matched Traits (30 mins)
**Objective:** Enhance the Explanation Facility.

Currently, `CareerRecommendation` reports how many traits matched (`matched_traits`), but does not display *which* specific traits matched.

**Task:**
1. Update `CareerRecommendation` in `inference_engine.py` to include:
   ```python
   matched_trait_names: List[str]
   ```
2. In `InferenceEngine.evaluate()`, keep track of each condition that successfully matched the user's profile and save it to `matched_trait_names`.
3. In `app.py`, update the UI card to display the matching traits as green tags:
   ```python
   st.markdown("**Matched Traits:** " + ", ".join(rec.matched_trait_names))
   ```

---

### Exercise 4: Advanced Discussion — Rule Weighting (15 mins)
Currently, all 5 traits have equal weight ($20\%$ each).
- Discuss: Is *Education Background* as flexible as *Primary Interest*?
- How could the formula be modified to give higher weight to *Strongest Skill* ($30\%$) and lower weight to *Core Value* ($10\%$)?

---
