source ./path.sh
build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_phone_ug.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_phone_bg.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_train_ug.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_train_bg.ilm.gz
build-lm.sh -i data/local/dict/lm_test.text -n 1 -o data/local/lm_tmp/lm_test_ug.ilm.gz
build-lm.sh -i data/local/dict/lm_test.text -n 2 -o data/local/lm_tmp/lm_test_bg.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 1 -o data/local/lm_tmp/lm_dev_ug.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 2 -o data/local/lm_tmp/lm_dev_bg.ilm.gz