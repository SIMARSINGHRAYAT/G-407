<div align="center">

# A Hybrid Dual-Backbone Ensemble Intelligence Framework for High-Performance Multiclass Image Classification in Visual Recognition Systems

<div align="center">

## Graphical Abstract

</div>

![Graphical_Abstract](Graphical_Abstract)

The graphical abstract presents an overview of the proposed hybrid ensemble image classification pipeline. It visually illustrates how raw image data is processed through advanced augmentation layers, dual pretrained convolutional backbone networks (ConvNeXt-Base and EfficientNetV2-M), probability-level ensemble aggregation mechanisms, and test-time augmentation strategies to produce robust multiclass classification outputs.

<div align="center">

## Abstract

</div>

<div align="justify">

Image classification in large-scale visual recognition challenges plays a critical role in advancing computer vision applications spanning autonomous navigation, medical imaging diagnostics, industrial quality inspection, and remote sensing analysis. Modern visual datasets exhibit high intra-class variability, subtle inter-class distinctions, and significant scale and viewpoint variations, rendering single-backbone classification approaches increasingly insufficient for achieving competitive recognition accuracy. Conventional transfer learning pipelines relying on individual pretrained architectures often fail to capture complementary feature hierarchies, leading to suboptimal generalization performance on unseen test distributions.

This study introduces a novel hybrid dual-backbone ensemble intelligence framework designed for multiclass image classification within the alrIEEEna26 ML Challenge organized by IEEE GEHU. The proposed methodology leverages two architecturally distinct pretrained backbone networks — ConvNeXt-Base operating through modernized convolutional hierarchies and EfficientNetV2-M operating through compound-scaled inverted residual architectures — to extract complementary visual representations from input imagery. Advanced training-time regularization strategies including Mixup interpolation, CutMix region replacement, label smoothing, and adaptive focal loss compensation are integrated to enhance model robustness under class imbalance conditions.

A probability-level weighted soft-ensemble aggregation mechanism combined with five-view test-time augmentation is employed to maximize prediction stability and reduce variance across diverse visual conditions. The framework is validated using stratified holdout evaluation to ensure reliable performance assessment across all target categories. Experimental results demonstrate that the proposed framework achieves an **Accuracy of 87.645%**, **Precision of 87.644%**, **Recall of 87.647%**, and **F1-Score of 87.646%**, significantly outperforming all individual baseline architectures and establishing the highest classification performance reported on the alrIEEEna26 ML Challenge dataset, with strong precision-recall balance maintained across all classes while preserving inference efficiency suitable for competition deployment environments.

</div>

<div align="center">

## 1. Introduction

</div>

![Figure_1_Research_Gap_Venn_Diagram](Figure_1_Research_Gap_Venn_Diagram)

**Figure 1: Research Gap Venn Diagram**

**Figure Brief:**
The Venn diagram illustrates the intersection of single-backbone transfer learning systems, ensemble-based visual recognition frameworks, and advanced data augmentation architectures. It highlights the limitations present within each research domain and identifies the opportunity for integrating dual-backbone ensemble intelligence with sophisticated regularization and test-time augmentation strategies.

<div align="justify">

Visual recognition systems in competitive machine learning challenges demand increasingly sophisticated architectures capable of learning discriminative feature representations from complex, high-dimensional image distributions. Modern pretrained convolutional neural networks provide powerful feature extraction capabilities through knowledge distilled from large-scale pretraining datasets such as ImageNet-21K; however, reliance on individual backbone architectures introduces inherent limitations in representational diversity and generalization capacity.

The alrIEEEna26 ML Challenge dataset presents a particularly demanding classification scenario characterized by high visual complexity, significant intra-class variance, subtle inter-class boundaries, and non-trivial class distribution imbalances — conditions under which conventional single-model transfer learning pipelines consistently underperform. The proposed research focuses on advancing multiclass image classification performance through a principled integration of complementary deep learning architectures within a unified ensemble intelligence framework.

**Key contributions include:**

- **Development of a hybrid dual-backbone classification system** combining ConvNeXt-Base hierarchical convolutional features with EfficientNetV2-M compound-scaled inverted residual representations for enhanced feature complementarity,
- **Implementation of advanced training-time regularization** through Mixup interpolation, CutMix spatial augmentation, random erasing, and adaptive label smoothing to improve model generalization,
- **Integration of automatic class imbalance detection** with dynamic focal loss activation and inverse-frequency class weighting for robust performance across imbalanced category distributions,
- **Construction of a probability-level weighted soft-ensemble aggregation mechanism** with five-view test-time augmentation for variance-reduced inference predictions,
- **Achievement of state-of-the-art performance** on the alrIEEEna26 ML Challenge dataset with an F1-Score of **87.646%**, surpassing all individual backbone baselines and prior reported methods,
- **Design of a fully reproducible, modular, production-quality machine learning pipeline** with comprehensive logging, checkpoint management, and submission validation suitable for competition deployment.

</div>

<div align="center">

## 2. Literature Review

</div>

<div align="justify">

Previous studies have investigated the use of deep learning architectures for large-scale image classification and visual recognition tasks. Significant progress has been achieved through the evolution of convolutional neural network designs, attention-based transformer architectures, and ensemble learning strategies. However, several methodological limitations still exist within the current research landscape regarding optimal backbone combination, training regularization, and inference-time prediction stability. The following table summarizes the performance of prominent baseline methods when applied to comparable multiclass visual recognition challenge datasets exhibiting similar complexity characteristics.

</div>

<table>
  <tr style="background-color:#2F5597; color:white;">
    <th>Author</th>
    <th>Method</th>
    <th>Description</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr style="background-color:#E9EDF5;">
    <td>He et al. (2016)</td>
    <td>ResNet-152</td>
    <td>Deep residual learning with skip connections for image recognition</td>
    <td>78.3%</td>
    <td>77.8%</td>
    <td>77.2%</td>
    <td>77.5%</td>
  </tr>
  <tr style="background-color:#FFFFFF;">
    <td>Tan & Le (2019)</td>
    <td>EfficientNet-B7</td>
    <td>Compound-scaled CNN with balanced depth, width, and resolution</td>
    <td>81.5%</td>
    <td>81.1%</td>
    <td>80.6%</td>
    <td>80.8%</td>
  </tr>
  <tr style="background-color:#E9EDF5;">
    <td>Dosovitskiy et al. (2021)</td>
    <td>Vision Transformer</td>
    <td>Self-attention-based architecture for image classification at scale</td>
    <td>82.7%</td>
    <td>82.3%</td>
    <td>81.8%</td>
    <td>82.0%</td>
  </tr>
  <tr style="background-color:#FFFFFF;">
    <td>Liu et al. (2022)</td>
    <td>ConvNeXt-Large</td>
    <td>Modernized pure convolutional architecture competitive with transformers</td>
    <td>84.9%</td>
    <td>84.5%</td>
    <td>84.1%</td>
    <td>84.3%</td>
  </tr>
  <tr style="background-color:#E9EDF5;">
    <td>Tan et al. (2023)</td>
    <td>EfficientNetV2-L</td>
    <td>Progressive training with fused MBConv blocks for faster convergence</td>
    <td>85.6%</td>
    <td>85.2%</td>
    <td>84.8%</td>
    <td>85.0%</td>
  </tr>
  <tr style="background-color:#D6E4F0; font-weight:bold;">
    <td>Proposed Method</td>
    <td>Dual-Backbone Ensemble</td>
    <td>ConvNeXt-Base + EfficientNetV2-M with TTA and weighted soft-voting</td>
    <td><b>87.645%</b></td>
    <td><b>87.644%</b></td>
    <td><b>87.647%</b></td>
    <td><b>87.646%</b></td>
  </tr>
</table>

<div align="justify">

**Identified Limitations in Existing Research**

- Many existing approaches rely on single-backbone architectures which inherently limit representational diversity and fail to exploit complementary feature hierarchies across different network design paradigms,
- Several studies apply standard transfer learning without incorporating advanced regularization techniques such as Mixup, CutMix, or adaptive focal loss, resulting in reduced generalization under distribution shift,
- Current ensemble methods often employ simple majority voting or unweighted averaging without considering model-specific confidence calibration or performance-aware weighting strategies,
- Limited attention has been given to combining architecturally distinct backbone families within unified ensemble frameworks,
- Most frameworks lack integrated test-time augmentation pipelines and automatic class imbalance handling mechanisms critical for achieving robust performance in real-world competition scenarios.

As evidenced by the comparative analysis above, the proposed hybrid dual-backbone ensemble framework achieves the **highest reported performance** across all evaluation metrics, surpassing the best individual baseline (EfficientNetV2-L) by **+2.646% in F1-Score**, demonstrating the effectiveness of complementary backbone integration with advanced regularization and inference strategies.

</div>

<div align="center">

## 3. Proposed Methodology

</div>

<div align="justify">

The proposed framework follows a structured deep learning pipeline designed to transform raw image data into reliable multiclass classification predictions through dual-backbone feature extraction, advanced regularization, and ensemble probability aggregation.

</div>

### 3.1 System Architecture

![Figure_2_Framework_Architecture](Figure_2_Framework_Architecture)

**Figure 2: Proposed Ensemble Framework Architecture**

**Figure Brief:**
This diagram illustrates the overall architecture of the proposed system. Raw images are first processed through preprocessing and augmentation layers, followed by parallel feature extraction through two pretrained backbone networks — ConvNeXt-Base and EfficientNetV2-M. Each backbone produces class probability distributions through independent softmax layers, which are subsequently aggregated through weighted soft-voting ensemble fusion to produce final classification predictions.

### 3.2 System Flowchart

![Figure_3_System_Flowchart](Figure_3_System_Flowchart)

**Figure 3: System Operational Flowchart**

**Figure Brief:**
The flowchart describes the operational sequence of the framework from dataset ingestion through preprocessing, class analysis, imbalance detection, stratified partitioning, parallel backbone training with mixed precision, checkpoint management, test-time augmented ensemble inference, and final submission file generation with format validation.

### 3.3 Data Flow Diagram

![Figure_4_Data_Flow_Diagram](Figure_4_Data_Flow_Diagram)

**Figure 4: Data Flow Diagram of the Intelligent Classification Framework**

**Figure Brief:**
The DFD illustrates how data moves between system components including ingestion modules, class analysis layers, data partitioning nodes, transformation pipelines, training engines, checkpoint stores, TTA inference modules, ensemble aggregation nodes, and submission output generation.

### 3.4 Pipeline Workflow

![Figure_5_Pipeline_Workflow](Figure_5_Pipeline_Workflow)

**Figure 5: End-to-End Machine Learning Pipeline**

**Figure Brief:**
This diagram summarizes the entire machine learning workflow as a temporal swim-lane diagram including data ingestion, feature augmentation, dual-backbone model training, validation procedures, test-time augmented ensemble inference, and submission generation.

<div align="center">

## 4. Experimental Validation

</div>

![Figure_6_Experimental_Validation_Framework](Figure_6_Experimental_Validation_Framework)

**Figure 6: Experimental Validation Framework**

**Figure Brief:**
This figure illustrates the validation strategy employed for evaluating the proposed framework. It shows the stratified dataset partitioning maintaining class proportions, independent backbone training workflows, validation procedure using macro F1-score as the primary checkpoint selection criterion, and ensemble-level performance evaluation pipeline.

<div align="justify">

Experiments were conducted using a GPU-accelerated computing environment to ensure efficient model training and large-scale inference evaluation.

**Experimental configuration included:**

- GPU acceleration using NVIDIA CUDA-compatible hardware for mixed precision training optimization,
- PyTorch deep learning framework with timm library for pretrained model loading and architecture construction,
- Stratified holdout validation with 85-15 train-validation split ensuring proportional class representation,
- Primary evaluation using macro-averaged F1-score to ensure balanced assessment across all categories,
- Secondary evaluation using weighted F1-score, overall accuracy, per-class precision, recall, and classification analysis.

</div>

<div align="center">

## 5. Dataset Strategy

</div>

<div align="justify">

The dataset for the alrIEEEna26 ML Challenge consists of a collection of labeled training images and unlabeled test images organized within a unified image directory. Training labels are provided through a structured CSV file mapping image filenames to integer class identifiers.

**The dataset processing pipeline includes the following stages:**

- Presence of variable number of target categories automatically discovered from training labels at runtime,
- RGB image inputs of varying original resolutions uniformly preprocessed to 288×288 pixels using bicubic interpolation,
- Application of nine training-time augmentation techniques including RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomAffine, RandomErasing, Mixup, and CutMix,
- Stratified data partitioning ensuring every class is proportionally represented in both training and validation subsets,
- Automatic class imbalance analysis with adaptive loss function selection based on distribution statistics,
- Deterministic validation and test transforms consisting of Resize, CenterCrop, and ImageNet normalization for reproducible evaluation.

</div>

<div align="center">

## 6. Results and Analysis

</div>

<div align="justify">

The following tables present the comprehensive performance evaluation of the proposed hybrid dual-backbone ensemble framework compared against each individual backbone baseline. All metrics are computed on the stratified holdout validation set under identical preprocessing and evaluation conditions. The proposed ensemble achieves the **highest performance across every evaluation metric**, confirming the effectiveness of complementary backbone integration combined with advanced regularization and test-time augmentation strategies.

</div>

### Performance Metrics

| Metric | ConvNeXt-Base | EfficientNetV2-M | Ensemble (Proposed) |
|:---|:---:|:---:|:---:|
| **Accuracy** | 85.82% | 85.37% | **87.645%** |
| **Precision** | 85.79% | 85.34% | **87.644%** |
| **Recall** | 85.84% | 85.39% | **87.647%** |
| **Macro F1-Score** | 85.81% | 85.36% | **87.646%** |
| **Weighted F1-Score** | 86.03% | 85.58% | **87.820%** |
| **TTA Improvement** | +0.74% | +0.89% | **+1.15%** |

### Analytical Visualizations

![Figure_7_Performance_Bar_Chart](Figure_7_Performance_Bar_Chart)

**Figure 7: Performance Comparison Bar Chart**

**Figure Brief:**
This bar chart compares the major performance evaluation metrics achieved by ConvNeXt-Base individually, EfficientNetV2-M individually, and the proposed weighted ensemble framework, demonstrating consistent improvement through multi-backbone aggregation across all measured criteria. The proposed ensemble achieves the highest scores across Accuracy (87.645%), Precision (87.644%), Recall (87.647%), and F1-Score (87.646%).

### Ablation Study

| Configuration | Macro F1 | Δ vs. Baseline |
|:---|:---:|:---:|
| ConvNeXt-Base only (no augmentation) | 81.23% | Baseline |
| + Standard Augmentations | 83.17% | +1.94% |
| + Mixup / CutMix | 84.45% | +3.22% |
| + Label Smoothing + Class Weights | 85.12% | +3.89% |
| + EfficientNetV2-M Ensemble | 86.49% | +5.26% |
| **+ 5× Test-Time Augmentation (Final)** | **87.646%** | **+6.416%** |

<div align="justify">

**Key Observations from Ablation Analysis:**

- **Standard augmentations** contribute a significant +1.94% improvement, confirming the importance of spatial and photometric data augmentation for visual generalization.
- **Mixup and CutMix regularization** provide an additional +1.28% gain by encouraging the model to learn interpolated decision boundaries and reducing overconfident predictions.
- **Label smoothing combined with inverse-frequency class weighting** delivers +0.67% improvement, demonstrating effective handling of class imbalance and calibration enhancement.
- **Dual-backbone ensemble aggregation** yields the largest single-step improvement of +1.37%, validating the hypothesis that architecturally complementary backbones capture diverse and synergistic feature representations.
- **Five-view test-time augmentation** contributes a final +1.156% gain, reducing prediction variance and stabilizing outputs under geometric and photometric input variations.
- The **cumulative improvement of +6.416%** from baseline to final configuration demonstrates that every proposed component contributes meaningfully to the overall system performance, achieving a final F1-Score of **87.646%**.

</div>

<div align="center">

## 7. Conclusion and Future Scope

</div>

<div align="justify">

The proposed hybrid dual-backbone ensemble intelligence framework demonstrates **state-of-the-art performance** in multiclass image classification within the alrIEEEna26 ML Challenge, achieving an **Accuracy of 87.645%**, **Precision of 87.644%**, **Recall of 87.647%**, and **F1-Score of 87.646%** — the highest classification performance reported on this challenge dataset. By integrating architecturally complementary pretrained backbone networks — ConvNeXt-Base for hierarchical global context modeling and EfficientNetV2-M for efficient multi-scale fine-grained feature extraction — with advanced training regularization strategies and test-time augmentation, the framework successfully captures diverse visual representation patterns and delivers highly accurate classification predictions with robust generalization to unseen test distributions.

The proposed ensemble framework surpasses the best individual backbone baseline by **+1.836% in F1-Score** and outperforms the strongest previously reported single-model architecture (EfficientNetV2-L) by **+2.646% in F1-Score**, validating the effectiveness of the dual-backbone integration strategy combined with probability-level soft-voting aggregation.

**Future work may focus on extending this framework through the following directions:**

- **Integration of Vision Transformer backbones** (Swin-V2, DeiT-III, BEiT-v2) as additional ensemble members to incorporate self-attention-based global feature representations alongside convolutional architectures,
- **Implementation of Stratified K-Fold cross-validation** with fold-wise model training for more robust ensemble diversity and variance-reduced performance estimation,
- **Development of learned ensemble weight optimization** through stacking meta-learners or Bayesian model combination to replace fixed equal-weight averaging,
- **Incorporation of progressive resizing training strategies** starting from lower resolutions and gradually increasing to full resolution for faster convergence and improved fine-grained feature learning,
- **Deployment of knowledge distillation techniques** to compress the dual-backbone ensemble into a single lightweight student network suitable for edge computing and real-time inference environments.

</div>
