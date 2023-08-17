source ./path.sh
compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz
compile-lm data/local/lm_tmp/lm_train_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_train_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_bg.arpa.gz
compile-lm data/local/lm_tmp/lm_test_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_test_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_test_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_test_bg.arpa.gz
compile-lm data/local/lm_tmp/lm_dev_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_dev_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_dev_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_dev_bg.arpa.gz