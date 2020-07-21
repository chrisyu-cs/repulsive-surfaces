ffmpeg -i $1 -vf drawtext="fontfile=palatino.ttf: \
text='$2': fontcolor=#ffffff: fontsize=32: box=1: boxcolor=black@0.5: \
boxborderw=10: x=20: y=(h-text_h)-20" -codec:a copy $3
