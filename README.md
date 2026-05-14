[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/odUTfyBX)
# AI Stats Lab: Bias-Variance Tradeoff

## Topic

This lab focuses on the **bias-variance tradeoff** in machine learning.

You will work with:

- Nonlinear data generation
- Polynomial regression
- Model complexity
- Train/dev error comparison
- Bias and variance diagnosis
- Regularization comparison
- Model improvement recommendations

The main idea is that model complexity affects generalization.

A very simple model may **underfit** and have **high bias**.

A very complex model may **overfit** and have **high variance**.

A good model balances bias and variance.

This lab is connected to model iteration: instead of randomly trying model changes, we use train/dev error to decide whether to increase model complexity, add regularization, collect more data, or keep the model.

---

## Repository Structure

Your repository must contain exactly these files:

```text
AI_stats_lab.py
test_AIstats_lab.py
README.md
