#Training data selection for Social Media NER

## Environment 
pytorch==1.7.1
seqeval==0.0.12

## Obtain the glove and bert embedding for a dataset
### For glove, run
python data_selection/read_glove_vector
### For bert, run
sh run_obtain_sent_repre.sh

## Generate the sentence id ranking
### For coverage-based method, run
python data_selection/generate_coverage_based_idx.py
### For vector-based method, run
python data_selection/generate_vector_base_idx.py

## for NER training with TDS
### Run four shell files to get the sota results reported in paper, and you can change the --train_data_dir to select different proporthion source data.
sh run_TDS_btw.sh
sh run_TDS_twitter.sh
sh run_TDS_w16.sh
sh run_TDS_w17.sh

## The parameters need to change
python run_seperate_train_with_domain_encode.py \
--train_data_dir=source_train_data_dir \
--dev_data_dir=target_dev_data_dir \
--test_data_dir=target_test_data_dir \
--target_train_data_dir=target_train_data_dir \
--target_label_list PER ORG LOC \
--source_label_list PER ORG LOC MISC \
--bert_model=the path to bert-large-cased \
--task_name=ner \
--output_dir=output_path \
--max_seq_length=128 \
--num_train_epochs 10 \
--do_train \
--gpu_id 2 \
--learning_rate 5e-5 \
--warmup_proportion=0.1 \
--train_batch_size=32

You need to change all paths to your own path, and the target_label_list and source_label_list need to be modified according to the dataset.

## utils folder contains python files for data processing and visualization.
