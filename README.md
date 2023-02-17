# Source Code
  ## Final Evaluation and Demo.ipynb
      - The Colab notebook used to evaluate and demo some translations for all models we experimented with
  ## data
     - This directory contains the data used in our project
        + original: unprocessed data
        + processed: processed data
        + translations: inference files for the test set, used for evaluation
  ## model
       - This directory contains all the trained models in this project
       - However, our models are too heavy to upload on Teams. 
         Our models can be found here: https://drive.google.com/drive/folders/10VKcZ2QEHmrHzKgsoflFlmREj37Xsgw3?usp=sharing
         
  ## src
       -  This directory contains all source codes in our project
          + preprocess.py: code to preprocess data
          + dataset.py: code to build dataset and dataloader
          + vocabulary.py:  code to build vocabulary
          + model.py: code for our model
          + Translator.py: API to train and perform prediction for our models
          + Training Notebooks: contains Colab notebooks to train our models
          + Augmentation: contains source code for augmentation techniques
  ## utils
       - bpe: contains BPE files
       - config: contains configuration files for each models
       - runs: contains val_loss and learning_rate information 
       - vocab: contains vocabularies for each models
   
     
     