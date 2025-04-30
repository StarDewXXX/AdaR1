The official repository of paper "AdaR1: From Long-Cot to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization"

## 🔍 AdaR1: Adaptive and Efficient Reasoning for Large Language Models

**AdaR1** is a two-stage adaptive reasoning framework designed to improve the efficiency of large language models (LLMs) without compromising their reasoning capabilities. While Long Chain-of-Thought (Long-CoT) reasoning has shown strong performance on complex tasks, it often incurs substantial inference overhead and does not always lead to better accuracy across all problems.

To address this, AdaR1 introduces a flexible strategy that dynamically adjusts the depth of reasoning based on problem difficulty and redundancy. 

### 🚀 Highlights
- A **Hybrid Adaptive Reasoning Model** that can adaptively generate Long-CoT or Short-CoT according to the difficulty of problem.
- Achieve **50%+ reduction in average reasoning length**, significantly lowering inference cost
- Achieve **Preserved accuracy** on five mathematical reasoning benchmarks

### 📦 Code & Model Release

The implementation and model checkpoint will be open-sourced soon.

## Results of AdaR1-7B
<img src="figures/adar1-table.png"></img>

## Accuracy of Different Methods
<img src="figures/acc1.jpg"></img>

## Tokens Used of Different Methods
<img src="figures/token1.jpg"></img>
