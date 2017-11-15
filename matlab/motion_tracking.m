dbstop if error;

WINDOW_SIZE = 13;

% 1. Read images
old_frame = imread('../assets/checkerboard_1.jpg');
new_frame = imread('../assets/checkerboard_2.jpg'); 

% 2. Detect corners
corners = get_good_features(rgb2gray(old_frame),  30, 13, WINDOW_SIZE);
% corners = [
%     527,498;
%     534,891;
%     733,1079;
%     733,1082;
%     760,1274;
%     760,1275;
%     1116,1461;
%     895,1463;
%     754,1465;
%     755,1465;
%     543,1466;
%     1304,1546;
%     1304,1548;
%     1493,1647;
%     1117,1650;
%     929,1651;
%     929,1652;
%     1118,1837;
%     1119,1838;
%     851,1840;
%     870,1840;
%     872,1840;
%     897,1840;
%     899,1840;
%     922,1840;
%     924,1840;
%     780,1841;
%     781,1841;
%     797,1841;
%     837,1841
%     ];
% show_corners(old_frame, corners, WINDOW_SIZE);
[tracked_corners, st, err] = lucas_kanade(rgb2gray(old_frame), rgb2gray(new_frame), corners, WINDOW_SIZE);
subplot(1,1,1), imshow(new_frame);
hold on;
good_old = corners(st == 1, :);
good_new = tracked_corners(st == 1, :);

for i = 1 : size(good_old, 1)
    old_r = good_old(i,1);
    old_c = good_old(i,2);
    
    new_r = good_new(i,1);
    new_c = good_new(i,2);
    
    extended_old_r = old_r - (new_r - old_r) * 20;
    extended_old_c = old_c - (new_c - old_c) * 20;
    
    line_c = [extended_old_c, new_c];
    line_r = [extended_old_r, new_r];
    line(line_c, line_r, 'Color', 'green', 'LineWidth', 2);
    
    d = floor(WINDOW_SIZE / 2);
    rectangle('Position', [new_c-d, new_r-d, WINDOW_SIZE, WINDOW_SIZE], ...
        'FaceColor', [1, 0, 0]);
end
hold off;
