if [ -e $1 ]
  then
    rm -r $1
fi

mkdir $1
cp IN mc.exe $1
cp *.py $1
cp *.data $1 

cd $1
./mc.exe | tee log 
