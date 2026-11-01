function coefficients = runmfccCustom(frames, win, DHTmat, filterBank, DCTmatrix)
    % coefficients = log2(((frames .* win) * DHTmat).^2 * filterBank) * DCTmatrix;
    coefficients = linLog2(((frames*2^11 .* win) * DHTmat/1323).^2*2^6 * filterBank,2^-8) * DCTmatrix;
end

function x = linLog2(x,s)
    offsets = 2^8*[-0.0774574 -0.540415 -1.10673 100.27592]; % these offsets do not change even with scaling
    amplitude = s*[101.66462 8.33727 0.539039 0.0338265]; % pre scaling amplitude (is the slope of the line and effects offsets)
    cutoffs = [9.24151 128.32504 2021.07345]; % separations between bands
    div2 = x>cutoffs(2);
    band4 = x>cutoffs(3) & div2;
    band3 = x<=cutoffs(3) & div2;
    band2 = x>cutoffs(1) & ~div2;
    band1 = x<=cutoffs(1) & ~div2;
    x(band1) = amplitude(1)*(x(band1)+offsets(1));
    x(band2) = amplitude(2)*(x(band2)+offsets(2));
    x(band3) = amplitude(3)*(x(band3)+offsets(3));
    x(band4) = amplitude(4)*(x(band4)+offsets(4));
end