#!/usr/bin/zsh

i=1
epoch_per_fold=2
total_fold=1
j=$epoch_per_fold
f=0
grep Vali step2.log > result.log
echo "#!/bin/bash" > .run_evaluate.sh

while true
do
   #echo $i $j
   value=`sed -n "$i,$j"p result.log|sed s'/.*=//g'|uniq|sort|tail -1`
   epoch=`sed -n "$i,$j"p result.log|grep $value|sed 's/.*Epoch\[//g'|sed 's/].*//g'`
   ((epoch=epoch+1))
   sed -n "$i, $j"p result.log| grep $value |sed "s/.*Vali/fold $f, epoch: $epoch Vali/g"
   echo "./utils/prepare_image.py --fold_id $f --epoch $epoch --threshold 0 --file_list './finalva.lst.processed'" >> .run_evaluate.sh
   ((i=i+epoch_per_fold))
   ((j=j+epoch_per_fold))
   ((f=f+1))
   if ((j>epoch_per_fold*total_fold))
   then
       break
   fi
done

echo "./utils/combine_evaluate.py --file_list='./finalva.lst' --compute_dcs=1" >> .run_evaluate.sh

chmod +x ./.run_evaluate.sh
./.run_evaluate.sh
