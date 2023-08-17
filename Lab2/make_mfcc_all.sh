# done
#!/bin/bash
source ./path.sh
source ./conf/mfcc.conf

utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

./steps/make_mfcc.sh ./data/train
./steps/make_mfcc.sh ./data/test
./steps/make_mfcc.sh ./data/dev

cp data/train/.backup/feats.scp data/train/feats.scp
cp data/test/.backup/feats.scp data/test/feats.scp
cp data/dev/.backup/feats.scp data/dev/feats.scp

./steps/compute_cmvn_stats.sh ./data/train ./exp/make_mfcc/train mfcc
./steps/compute_cmvn_stats.sh ./data/test ./exp/make_mfcc/test mfcc
./steps/compute_cmvn_stats.sh ./data/dev ./exp/make_mfcc/dev mfcc
