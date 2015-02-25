#ffmpeg -f image2 -r 60 -i warped_stereo%04d.jpg -vcodec mpeg4 -y warped_movie.mp4
#ffmpeg -r 60 -i warped_stereo%04d.jpg -vcodec mpeg4 -y warped_movie.mp4
ffmpeg -r 60 -i warped_stereo%04d.jpg -c:v libx264 -y warped_movie.mp4
