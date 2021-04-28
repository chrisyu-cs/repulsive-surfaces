dir=`dirname $1`
echo "Using base directory ${dir}"
for i in `cat $1`
do
	fname="${dir}/${i}"
	echo "Running ${fname}..."
	./build/bin/rsurfaces $fname --autolog --threads 1
done
