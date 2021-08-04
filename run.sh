#!/bin/bash
key_word='damage destroy strike bomb attack invade battle stab hijack strangl protest retreat surrender'
#key_word='House White'
for keyword in $key_word
do
./run_srl.sh $keyword &
done