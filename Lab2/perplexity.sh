source ./path.sh
source ./cmd.sh

compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz --eval=data/local/dict/lm_test.text --dub=10000000
compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz --eval=data/local/dict/lm_test.text --dub=10000000
compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz --eval=data/local/dict/lm_dev.text --dub=10000000
compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz --eval=data/local/dict/lm_dev.text --dub=10000000