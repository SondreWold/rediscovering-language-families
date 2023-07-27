rm -r models
rm -r parsed_corpora
mkdir models
mkdir parsed_corpora

for i in "lt Lithuanian" "ee Ewe" "wo Wolof" "xh Xhosa" "zu Zulu" "fi Finnish" "et Estonian" "hu Hungarian" "es Spanish" "de German" "ru Russian" "is Icelandic" "he Hebrew" "ar Arabic" "shi Tachelhit" "so Somali" "wal Wolaytta"
do
    set -- $i
    echo Starting preprocessing of $2
    echo Parsing source corpora...
    python3 parse_source.py --code $1 --name $2
    echo Training word2vec model...
    python3 train_word2vec.py --corpus parsed_corpora/$2 --epochs 20 --window 5
    echo Converting word2vec model to torch format
    python3 convert.py $2
    rm models/$2.model
done