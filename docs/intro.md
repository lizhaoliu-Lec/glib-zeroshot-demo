["**GLIP: Grounded Language-Image Pre-training. CVPR 2022, Best Paper Finalist**"](https://arxiv.org/abs/2112.03857)

This is the HuggingFace Gradio Demo for GLIP. The model requires an image, and a text to be the inputs. The text input can either be a natural sentence description (grounding), or a simple concatenation of some random categories (object detection).

The paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich.

Code: https://github.com/microsoft/GLIP

**News**: We are also holding an ODinW challenge at [the CV in the Wild Workshop @ ECCV 2022](https://computer-vision-in-the-wild.github.io/eccv-2022/). We hope our open-source code encourage the community to participate in this challenge!
