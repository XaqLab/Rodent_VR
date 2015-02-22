#ffmpeg -f image2 -r 60 -i stereo%04d.jpg -vcodec mpeg4 -y movie.mp4
ffmpeg -r 60 -i stereo%04d.jpg -c:v libx264 -y movie.mp4
