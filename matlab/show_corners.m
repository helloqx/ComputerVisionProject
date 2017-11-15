function show_corners(frame, corners, window_size)
    % Show good corners
    subplot(1,1,1), imshow(frame);
    hold on;
    for i = 1 : size(corners, 1)
        r = corners(i, 1);
        c = corners(i, 2);

        d = floor(window_size / 2);
        rectangle('Position', [c-d, r-d, window_size, window_size], ...
            'FaceColor', [1, 0, 0]);
    end
    hold off;
end