ffmpeg -i $1 -i $2 -filter_complex "[1:v]format=argb,geq=r='r(X,Y)':a='0.25*alpha(X,Y)'[zork]; [0:v][zork]overlay" -vcodec libx264 $3
