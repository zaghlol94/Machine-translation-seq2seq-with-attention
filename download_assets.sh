python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
gdown "17ifILTP5iwnQHwX1e-2WVHFAuyTVnuqm"
mv seq2seq_with_attention.zip src/
cd src/
unzip seq2seq_with_attention.zip
rm seq2seq_with_attention.zip
