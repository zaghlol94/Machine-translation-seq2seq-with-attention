python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
gdown "1p_AUVEtxG_4ZJLoG8w-pJ1UgKy47NB8P"
mv seq2seq_phrase_rep.zip src/
cd src/
unzip seq2seq_phrase_rep.zip
rm seq2seq_phrase_rep.zip
