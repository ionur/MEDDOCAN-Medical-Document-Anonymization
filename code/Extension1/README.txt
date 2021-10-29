In order to see the performance of Extension1:

Run 
  python milestone_4_ext1.py

For evaluation:
Evaluation will use the script under the code directory.
Go to code/

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
  Total (156 docs)                   Precision      0.8951              
                                     Recall         0.8285              
                                     F1             0.8606              
  ------------------------------------------------------------
