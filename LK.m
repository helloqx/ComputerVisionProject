v = VideoReader('traffic.mp4');

% Step 1: Reading the two frames
frame1 = double(readFrame(v));
frame2 = double(readFrame(v));

f_width_o = v.Width;
f_height_o = v.Height;
