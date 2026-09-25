# Methodology

The agreed high-level pipeline is:

Raw records → Normalization → Candidate generation → Pairwise features →
Matching model → Probability scores → Threshold optimization → Final matches

Use grouped validation by Source1 entity. Candidate generation focuses on
recall; final matching focuses on precision. Evaluate with macro F0.5.
Change one major factor per experiment and validate locally before submitting
to the leaderboard. Reproduce a result using its experiment ID, parameters,
and Git commit.

## 1. Data Understanding

## 2. Normalization

## 3. Candidate Generation

## 4. Feature Engineering

## 5. Matching Model

## 6. Threshold Optimization

## 7. Evaluation

## 8. Final Inference
