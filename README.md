
# 🔍 Ada-R1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization

📢 **Update (Sep 2025):**
Our paper **\[Ada-R1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization]** has been **accepted at NeurIPS 2025** 🎉.

---

## ✨ Overview

**AdaR1** is a two-stage **adaptive reasoning framework** designed to improve the efficiency of large language models (LLMs) without sacrificing reasoning performance.

While **Long Chain-of-Thought (Long-CoT)** reasoning enhances LLMs on complex tasks, it often leads to **substantial inference overhead** and does not always guarantee higher accuracy.

To address this, AdaR1 introduces a **bi-level adaptive strategy** that dynamically controls reasoning depth by considering both **problem difficulty** and **reasoning redundancy**.

---

## 🚀 Key Contributions

* 🔄 **Hybrid Adaptive Reasoning**: Dynamically switches between **Long-CoT** and **Short-CoT** according to problem difficulty.
* ⚡ **Efficiency**: Reduces average reasoning length by **50%+**, cutting inference cost significantly.
* 🎯 **Accuracy Preservation**: Maintains accuracy across **five challenging mathematical reasoning benchmarks**.
* 🧠 **Bi-Level Optimization**: Introduces adaptive control at both *instance-level* and *token-level*.

---

## 📊 Results

### Results of AdaR1-7B

<img src="figures/adar1-table.png" width="600">

### Accuracy of Different Methods

<img src="figures/acc1.jpg" width="600">

### Tokens Used of Different Methods

<img src="figures/token1.jpg" width="600">

---

## 📦 Code & Model Release

The official implementation and model checkpoints will be **open-sourced soon**.
