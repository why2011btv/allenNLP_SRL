#!/bin/bash
FILES=/shared/why16gzl/allenNLP-SRL/nyt_file/section__2007_05_16_to_2007_08_13_src6742-manifest*
keyword=$1

for f in $FILES
do
  #echo "Processing $f file..."
  # take action on each file. $f store current file name
  # cat $f
  #for keyword in $key_word
  #do
  if grep -c $keyword $f
  then 
  echo "Processing $f file..."
  #mails=$(echo $f | tr "/")
  filename=$(basename "$f")
  echo $filename
  mkdir -p $keyword
  python3 allen_srl.py $f --output-file ./$keyword/$filename.srl
  fi
  #done
done