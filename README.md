# MEDDOCAN-Medical-Document-Anonymization
Bi-LSTM model which resulted in a 93% F1 score on the test set for competition organized by MEDDOCAN. Goal is to capture protected health information  in medical records to facilitate their anonymization 


For our extension 2, we have built a named entity recognizer which is often also considered as a sequence tagging task. We took reference from the paper [9]. The model architecture
12
involves Bi-LSTM and CRF. Additionally, it makes use of fasttext word embeddings for Spanish. It also builds word embeddings using character encodings. We have used word em- beddings(fasttext) concatenated with model word embeddings (char based Bi-LSTM) while training the model. We then extract contextual representation of each word in a given sen- tence by running Bi-LSTM on the sentence. In the end, we used CRF to decode the output and get the category of each word.
