dir=`dirname $1`
echo "Using base directory ${dir}"
for i in `cat $1`
do
	fname="${dir}/${i}"
	echo "Running ${fname}..."
	./bin/rsurfaces $fname --autolog
done