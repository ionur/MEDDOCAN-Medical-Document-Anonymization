# MEDDOCAN-Medical-Document-Anonymization
Bi-LSTM model which resulted in a 93% F1 score on the test set for competition organized by MEDDOCAN. Goal is to capture protected health information  in medical records to facilitate their anonymization 

Due to privacy constraints, clinical records with protected health information (PHI) cannot be directly shared. De-identification, i.e., the exhaustive removal, or replacement, of all mentioned PHI phrases has to be performed before making the clinical records available outside of hospitals. We have tried to identify PHI on medical records written in Spanish language.

In the project, we tried different approaches for the task of de-identification of PHI in Spanish clinical records. We started with simple rule-based model and then moved on to LinearSVC. We then tried a system that is a combination of rule-based, LinearSVC and static dictionaries of Spanish names and locations. Later, we trained a neural network that uses Bi-LSTM and CRF for named entity recognition. The neural model performed best for us on the given dataset.

For our extension 2, we have built a named entity recognizer which is often also considered as a sequence tagging task. We took reference from the paper [9]. The model architecture
12
involves Bi-LSTM and CRF. Additionally, it makes use of fasttext word embeddings for Spanish. It also builds word embeddings using character encodings. We have used word em- beddings(fasttext) concatenated with model word embeddings (char based Bi-LSTM) while training the model. We then extract contextual representation of each word in a given sen- tence by running Bi-LSTM on the sentence. In the end, we used CRF to decode the output and get the category of each word.
