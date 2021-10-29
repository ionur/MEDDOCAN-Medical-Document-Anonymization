################# PREPROCESSING #################
go to finalSubmission/code/

Training files -
    python preprocessing.py --dataDir ../data/train/gold/ --train

    After this pre-processing step, a pickle file - train_word_ner_startidx_dict.pickle - is created in the current
    working directory which has the data in the NER type tag format (BIO) which is used for building the model.

Dev files -
    python preprocessing.py --dataDir ../data/dev/gold/ --dev

    After this pre-processing step, a pickle file - dev_word_ner_startidx_dict.pickle - is created in the current
    working directory which has the data in the NER type tag format (BIO) which is used for building the model.

Test files -
    python preprocessing.py --dataDir ../data/test/gold/ --test

    After pre-processing step, a pickle file - test_word_ner_startidx_dict.pickle - is created in the current
    working directory which has the data in the NER type tag format (BIO) which is used for building the model.

################# GENERATE FILES NEEDED BY NEURAL MODEL #################
cd Extension2/
python create_vocabs.py --trainpickle ../train_word_ner_startidx_dict.pickle \
                        --devpickle ../dev_word_ner_startidx_dict.pickle \
                        --testpickle ../test_word_ner_startidx_dict.pickle \
                        --embfile wiki.es.vec --vocabEmbFile vocab_embeddings.npz


This would generate the following pickle files in the current working directory -
(i) chars.pickle
(ii) tags.pickle
(iii) words.pickle
(iv) vocab_embeddings.npz

################# RUN MODEL #################
# First make sure you have chars.pickle, tags.pickle, words.pickle, and vocab_embeddings.npz in Extension2/

mkdir Model     # model are saved here
mkdir plots     # confusion plots are saved here

cd Extension2/Code/
python train.py


################# EVALUATION #################
Go to finalSubmission/code/

Evaluation will use the script 'evaluate.py' under the code directory. Please make sure 'classes.py' and 'tags.py'
exist in the same folder.

1) Training evaluation:
  python evaluate.py brat ner ../data/train/gold ../data/train/system

2) Dev evaluation:
  python evaluate.py brat ner ../data/dev/gold ../data/dev/system

3)Test evaluation:
  python evaluate.py brat ner ../data/test/gold ../output/test/system

Report (SYSTEM: system):
------------------------------------------------------------
SubTrack 1 [NER]                   Measure        Micro
------------------------------------------------------------
Total (156 docs)                   Precision      0.9573
                                   Recall         0.9133
                                   F1             0.9348
------------------------------------------------------------
