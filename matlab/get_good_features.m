%{
Frame -> grayscale frame
max_corners -> number of corners wanted
min_distance -> for mosaicing
window_size -> gaussian window size

Return value: corners is in [rows, cols] format
%}
function corners = get_good_features(frame, max_corners, min_distance, window_size)
    EPSILON = 0.15;
    gkern = gausswin(window_size) * gausswin(window_size).';
    
    frame = double(frame);  % cast to avoid overflow
    [nrows, ncols] = size(frame);
    I_x = frame(2:nrows, 1:ncols-1) - frame(1:nrows-1, 1:ncols-1);
    I_y = frame(1:nrows-1, 2:ncols) - frame(1:nrows-1, 1:ncols-1);

    I_xx = I_x .* I_x;
    I_xy = I_x .* I_y;
    I_yy = I_y .* I_y;
    
    W_xx = conv2(I_xx, gkern);
    W_xy = conv2(I_xy, gkern);
    W_yy = conv2(I_yy, gkern);
    
    % Truncate W_xx, W_xy, W_yy to match pic's size
    hfw = floor(window_size / 2);  % half window_size
    [nrows, ncols] = size(I_xx);
    W_xx = W_xx(hfw+1:hfw+nrows, hfw+1:hfw+ncols);
    W_xy = W_xy(hfw+1:hfw+nrows, hfw+1:hfw+ncols);
    W_yy = W_yy(hfw+1:hfw+nrows, hfw+1:hfw+ncols);
    
    eig_mins = (W_xx .* W_yy - W_xy.^2) ./ (W_xx + W_yy + EPSILON);
    % Can replace with the faster method if needed (refer to py file)
    %{
    % This is too slow
    [nrows, ncols] = size(W_xx);
    eig_mins = zeros(nrows, ncols);
    for i = 1:nrows
        for j = 1:ncols
            W = [W_xx(i,j) W_xy(i,j); W_xy(i,j) W_yy(i,j)];
            eig_mins(i,j) = min(eig(W));
        end
    end
    %}
    
    max_eig_mins = zeros(nrows, ncols);
    for i = 1:min_distance:nrows
        for j = 1:min_distance:ncols
            i_end = min(i + min_distance-1, nrows);
            j_end = min(j + min_distance-1, ncols);
            
            window = eig_mins(i:i_end, j:j_end);
            [r, c] = find(ismember(window, max(window(:))));
            max_eig_mins(i + r-1, j + c-1) = window(r,c);
        end
    end
    
    desc_eig_mins = sort(max_eig_mins(:), 'descend');
    cut_off_eig_min = desc_eig_mins(max_corners);
    [rows, cols] = find(max_eig_mins >= cut_off_eig_min, max_corners);
    
    corners = [rows, cols];
end