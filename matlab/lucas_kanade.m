function [tracked_corners, status, err] = lucas_kanade(old_frame, new_frame, corners, window_size)
    D_THRESHOLD = 5e-2;

    tracked_corners = zeros(size(corners));
    status = zeros(size(corners, 1), 1);
    err = 0;
    
    gkern = gausswin(window_size) * gausswin(window_size).';
    
    old_frame = double(old_frame);  % cast to avoid overflow
    new_frame = double(new_frame);  % cast to avoid overflow
    [nrows, ncols] = size(old_frame);
    I_x = old_frame(2:nrows, 1:ncols-1) - old_frame(1:nrows-1, 1:ncols-1);
    I_y = old_frame(1:nrows-1, 2:ncols) - old_frame(1:nrows-1, 1:ncols-1);

    I_xx = I_x .* I_x;
    I_xy = I_x .* I_y;
    I_yy = I_y .* I_y;
    
    W_xx = conv2(I_xx, gkern);
    W_xy = conv2(I_xy, gkern);
    W_yy = conv2(I_yy, gkern);
    
    [nrows, ncols] = size(I_xx);
    W_xx = truncate_after_conv2(W_xx, window_size, nrows, ncols);
    W_xy = truncate_after_conv2(W_xy, window_size, nrows, ncols);
    W_yy = truncate_after_conv2(W_yy, window_size, nrows, ncols);
    
    I_minus_J = old_frame(1:nrows,1:ncols) - new_frame(1:nrows,1:ncols);
    I_minus_J_x = I_minus_J .* I_x;
    I_minus_J_y = I_minus_J .* I_y;
    W_I_minus_J_x = conv2(I_minus_J_x, gkern);  % get sum(I-J)Ix with Gaussian weight
    W_I_minus_J_y = conv2(I_minus_J_y, gkern);  % get sum(I-J)Iy with Gaussian weight
    
    [nrows, ncols] = size(I_minus_J_x);
    W_I_minus_J_x =  truncate_after_conv2(W_I_minus_J_x, window_size, nrows, ncols);
    W_I_minus_J_y =  truncate_after_conv2(W_I_minus_J_y, window_size, nrows, ncols);
    
    for i = 1 : size(corners, 1)
        r = corners(i, 1);
        c = corners(i, 2);

        Z = [W_xx(r, c), W_xy(r, c); W_xy(r, c), W_yy(r, c)];
        b = [W_I_minus_J_x(r, c); W_I_minus_J_y(r, c)];
        d = Z \ b;
        
        tracked_corners(i, 1) = r + d(1);
        tracked_corners(i, 2) = c + d(2);
        status(i) = dot(d, d) > D_THRESHOLD;
    end
end