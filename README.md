# UPV FAQ Semantic Text Similarity Test

Only supports fastText embeddings in .bin format

## Dependencies:

- [sentence_transformers](https://www.sbert.net/)
- [pandas](https://pandas.pydata.org)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [matplotlib](https://matplotlib.org)

## Usage:

    python3 faq_tests.py [--pretrained huggingface_model_name] [--questions question_dataset_path] [--answers answer_dataset_path] [--cmtime disp_time_seconds] [--cm] [--verb] [--save]
  
### Arguments:

- **model_path**: Path to the model to be evaluated

- **questions**: Path to spreadsheet file with questions

- **answers**: Path to spreadsheet file with answers

- **cmtime**: Cofusion matrix display duration in seconds

- **cm**: Show confusion matrices during evaluation

- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **save**: Save evaluation results, including optional confusion matrices, into an appropriately named 
folder next to the evaluated model

## Confusion inspector:

    python3 confusion_inspector.py [--pretrained huggingface_model_name] [--questions question_dataset_path] 

Shows a similarity heatmap between all dataset questions. Click on a specific pixel in the heatmap to print 
the corresponding pair of questions and their similarity value in the terminal.
