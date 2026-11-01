function coefficients = mfccCustom(x, fs, Q, opts)
    if nargin < 4
        [win, DHTmat, filterBank, DCTmat] = genmfccCustom(Q, fs);
    else
        [win, DHTmat, filterBank, DCTmat] = genmfccCustom(Q, fs, opts);
    end
    frames = windowSound(x, fs);
    coefficients = runmfccCustom(frames,win,DHTmat,filterBank,DCTmat);
end