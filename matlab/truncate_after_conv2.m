function M = truncate_after_conv2(M, window_size, nrows, ncols)
    hfw = floor(window_size / 2);  % half window_size
    M = M(hfw+1:hfw+nrows, hfw+1:hfw+ncols);
end