#!/usr/bin/zsh

i=1
epoch_per_fold=2
total_fold=1
j=$epoch_per_fold
f=0
grep Vali step2.log > result.log
echo "#!/bin/bash" > .run_test.sh

while true
do
   #echo $i $j
   value=`sed -n "$i,$j"p result.log|sed s'/.*=//g'|uniq|sort|tail -1`
   epoch=`sed -n "$i,$j"p result.log|grep $value|sed 's/.*Epoch\[//g'|sed 's/].*//g'`
   ((epoch=epoch+1))
   sed -n "$i, $j"p result.log| grep $value |sed "s/.*Vali/fold $f, epoch: $epoch Vali/g"
   echo "./prepare_image.py --fold_id $f --epoch $epoch --threshold 0 --file_list './test.lst.processed'" >> .run_test.sh
   ((i=i+epoch_per_fold))
   ((j=j+epoch_per_fold))
   ((f=f+1))
   if ((j>epoch_per_fold*total_fold))
   then
       break
   fi
done

threshold=`tail -1 combine_evaluate.log|sed "s/.*threshold //g"`
echo $threshold
echo "./combine_evaluate.py --file_list='./test.lst' --compute_dcs=0 --threshold=$threshold" >> .run_test.sh

chmod +x ./.run_test.sh
./.run_test.sh
