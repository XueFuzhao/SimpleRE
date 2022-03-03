export BERT_BASE_DIR=../bert_base

#python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin


python run_classifier.py   --task_name bertd  --do_train  --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 6   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir bert_f1  --gradient_accumulation_steps 1

python evaluate.py --f1dev bert_f1/logits_dev.txt --f1test bert_f1/logits_test.txt
