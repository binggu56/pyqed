if [ -e $1 ]
  then
    rm -r $1
fi

mkdir $1
cp IN qm.x $1
cp *.py $1
cp derivs.f90 $1
cd $1
./qm.x | tee log 
