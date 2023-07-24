FILES=$(git ls-files -m)

for f in $FILES
do
    scp -r -P 9696 $f samuel@143.106.23.200:mbrl/$f
done