# UPV FAQ Semantic Text Similarity Test

## Dependencies:

- [sentence_transformers](https://www.sbert.net/)
- [pandas](https://pandas.pydata.org)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [matplotlib](https://matplotlib.org)

## Usage:

    python3 faq_tests.py [--pretrained huggingface_model_name] [--questions question_dataset_path] [--answers answer_dataset_path] [--cmtime disp_time_seconds] [--save_dir output_directory] [--filename results_filename] [--cm] [--verb]
  
### Arguments:

- **pretrained**: Path to the model from the Huggingface website (in format username/modelname)
  
- **questions**: Path to spreadsheet file with questions

- **answers**: Path to spreadsheet file with answers

- **model_path**: Path to the model to be evaluated

- **cmtime**: Cofusion matrix display duration in seconds

- **cm**: Show confusion matrices during evaluation

- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **save_dir**: Directory, where results should be saved (Default: ./output/)

- **filename**: Name of the file with evaluation results. (Default: Accuracies.log)

## Confusion inspector:

    python3 confusion_inspector.py [--pretrained huggingface_model_name] [--questions question_dataset_path] 

Shows a similarity heatmap between all dataset questions. Click on a specific pixel in the heatmap to print 
the corresponding pair of questions and their similarity value in the terminal.
