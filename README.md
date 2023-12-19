# Interpretable-and-Reliable-Deep-learning

Objective:

Thoracic diseases like pneumonia, COPD and tuberculosis are a major health burden globally. Chest X-rays are critical for screening and diagnosis.
But physicians face challenges in accurately interpreting X-rays. AI can assist by providing a second opinion.
Our goal is to develop an AI system for multi-label classification of thoracic diseases from chest X-rays.
The system should be interpretable and indicate its precision boundaries to make it more trustworthy for physicians.

Implementation:

We will train a DenseNet CNN architecture on the CheXpert dataset of labeled chest X-ray images.
During training, we will use techniques like weighted sampling and early stopping to improve model precision on rare classes and provide tuning levers.
For model interpretation, we will generate saliency maps using Grad-CAM to highlight regions that influenced the prediction.
We will also train simple decision tree models on CNN features to provide rule-based explanations.
To visualize precision boundaries, we will apply conformal prediction to get prediction sets and confidence scores instead of just point estimates.

Applications:


The system can assist radiologists by providing a second opinion on chest X-ray diagnoses along with explanations.
Visualizing precision boundaries can inform physicians when to trust or verify the model's predictions.
It can be used as a screening tool in communities with limited access to radiologists.

Results:


We hope to achieve good accuracy with improved precision on most of the classes using the proposed techniques.
The interpretability methods will provide localization and rule-based explanations for predictions.
Conformal predictions will indicate cases where the model is uncertain and has lower estimated precision.


Extensions:


Applying transfer learning to test the model on other chest X ray datasets
Extending our idea to medical image segmentation
We can refine the interpretation methods to be more reflective of physician workflows and decision making. Techniques like counterfactual explanations can be explored.
The system can be developed into an interactive decision-support tool for radiologists.
