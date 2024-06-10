1. Preprocessed corpuses by normalization, standard preprocessing, removed (pause x.xx), extra whitespaces.
2. Removing entries with word count less than 3.
3. Filtered the data based on ratio, range between 0.75 to 1.25. Eg. I am trynna sleep <-> I am trying to sleep , 4 - 5 so ratio is 4/5= 0.8. 
4. AI corpus - started with 31130, filtered to 20810
5. Pro corpus - started with 5782, filtered to 4111


1. Did not exapand contracted words like ain't to ain 't while tokenization to preserve AAVE language features.
2. 



Datasets Creation 




Third Set:
Training Data for Set 3: Now selects 18k entries from ai_corpus and 3k from pro_corpus for training.
Validation Data for Set 3: Uses the remaining ai_corpus entries after training extraction for validation.
Test Data for Set 3: Uses the remaining pro_corpus entries after training extraction for testing.


    

python preprocess.py --source-lang aave --target-lang sae \
    --trainpref /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/train --validpref /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/valid --testpref /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/test \
    --destdir /N/u/saswar/Carbonate/AAVE/Fairseq/Set3_Preprocessed \
    --workers 20
    
    
On Set 3

INFO:fairseq_cli.preprocess:[aave] Dictionary: 27616 types
INFO:fairseq_cli.preprocess:[aave] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/train.aave: 21000 sents, 362619 tokens, 0.0% replaced (by <unk>)
INFO:fairseq_cli.preprocess:[aave] Dictionary: 27616 types
INFO:fairseq_cli.preprocess:[aave] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/valid.aave: 2810 sents, 51841 tokens, 4.63% replaced (by <unk>)
INFO:fairseq_cli.preprocess:[aave] Dictionary: 27616 types
INFO:fairseq_cli.preprocess:[aave] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/test.aave: 1111 sents, 11350 tokens, 3.65% replaced (by <unk>)
INFO:fairseq_cli.preprocess:[sae] Dictionary: 23600 types
INFO:fairseq_cli.preprocess:[sae] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/train.sae: 21000 sents, 343119 tokens, 0.0% replaced (by <unk>)
INFO:fairseq_cli.preprocess:[sae] Dictionary: 23600 types
INFO:fairseq_cli.preprocess:[sae] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/valid.sae: 2810 sents, 48775 tokens, 3.79% replaced (by <unk>)
INFO:fairseq_cli.preprocess:[sae] Dictionary: 23600 types
INFO:fairseq_cli.preprocess:[sae] /N/u/saswar/Carbonate/AAVE/Fairseq/Set3/test.sae: 1111 sents, 11312 tokens, 5.18% replaced (by <unk>)
INFO:fairseq_cli.preprocess:Wrote preprocessed data to /N/u/saswar/Carbonate/AAVE/Fairseq/Set3_Preprocessed






learn_bpe.py on SetA_Preprocessed/tmp/train.aave-sae...
100%|##################################################################################################| 15000/15000 [00:09<00:00, 1524.43it/s]
apply_bpe.py to train.aave...
apply_bpe.py to valid.aave...
apply_bpe.py to test.aave...
apply_bpe.py to train.sae...
apply_bpe.py to valid.sae...
apply_bpe.py to test.sae...
clean-corpus.perl: processing SetA_Preprocessed/tmp/bpe.train.aave & .sae to SetA_Preprocessed/train, cutoff 2-350, ratio 2
..
Input sentences: 22637  Output sentences:  22241
clean-corpus.perl: processing SetA_Preprocessed/tmp/bpe.valid.aave & .sae to SetA_Preprocessed/valid, cutoff 2-350, ratio 2

Input sentences: 228  Output sentences:  222
(nlp2) [saswar@g12 Fairseq]$ 

learn_bpe.py on SetB_Preprocessed/tmp/train.aave-sae...
100%|##################################################################################################| 10000/10000 [00:05<00:00, 1951.12it/s]
apply_bpe.py to train.aave...
apply_bpe.py to valid.aave...
apply_bpe.py to test.aave...
apply_bpe.py to train.sae...
apply_bpe.py to valid.sae...
apply_bpe.py to test.sae...
clean-corpus.perl: processing SetB_Preprocessed/tmp/bpe.train.aave & .sae to SetB_Preprocessed/train, cutoff 2-350, ratio 2
.
Input sentences: 10120  Output sentences:  9812
clean-corpus.perl: processing SetB_Preprocessed/tmp/bpe.valid.aave & .sae to SetB_Preprocessed/valid, cutoff 2-350, ratio 2

Input sentences: 102  Output sentences:  99
(nlp2) [saswar@g12 Fairseq]$ 

earn_bpe.py on SetC_Preprocessed/tmp/train.aave-sae...
100%|##################################################################################################| 15000/15000 [00:09<00:00, 1503.52it/s]
apply_bpe.py to train.aave...
apply_bpe.py to valid.aave...
apply_bpe.py to test.aave...
apply_bpe.py to train.sae...
apply_bpe.py to valid.sae...
apply_bpe.py to test.sae...
clean-corpus.perl: processing SetC_Preprocessed/tmp/bpe.train.aave & .sae to SetC_Preprocessed/train, cutoff 2-350, ratio 2
..
Input sentences: 23572  Output sentences:  23118
clean-corpus.perl: processing SetC_Preprocessed/tmp/bpe.valid.aave & .sae to SetC_Preprocessed/valid, cutoff 2-350, ratio 2

Input sentences: 238  Output sentences:  233

    

python preprocess.py --source-lang aave --target-lang sae \
    --trainpref /N/u/saswar/Carbonate/AAVE/Fairseq/SetC_Preprocessed/train --validpref /N/u/saswar/Carbonate/AAVE/Fairseq/SetC_Preprocessed/valid --testpref /N/u/saswar/Carbonate/AAVE/Fairseq/SetC_Preprocessed/test \
    --destdir /N/u/saswar/Carbonate/AAVE/Fairseq/SetC_Binarized \
    --workers 20