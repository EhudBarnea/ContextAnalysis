# ContextAnalysis
Understanding the utility of context for object detection

### Introduction

The detection of objects is mostly based on their local appearance without an explicit reasoning about their context. Seeking to exploit this meaningful source of information, many works have shown the ability of the context to predict object locations and improve their detection. However, in most cases (or object categories) the presented improvement was rather modest, leaving a gap between actual improvement and the expected one. In this project, we show why this is so and shed more light on the utility of context for object detection. Specifically, we provide a tool that calculates the best possible improvement that can be achieved by the inclusion of context, showing exactly when context can and cannot improve. With further analysis we provide a reason for the gap between achieved and expected improvement based on the inability of context to handle false detections due to localization errors, which are often abundant in different detectors and object categories. 

Barnea E. and Ben-Shahar O., On the Utility of Context (or the Lack Thereof) for Object Detection ([arXiv](http://arxiv.org/abs/1711.05471)).

### Folders

src - all code and implementation of the tools and analyses described in the paper.
lit_summary - the summary of detection improvement obtained by previous methods (including visualization tools).
