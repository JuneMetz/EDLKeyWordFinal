function [win, DHTmat, filterBank, DCTmat] = genmfccCustom(Q, fs, opts)
%MFCCCUSTOM Simplified clone of MATLAB's mfcc function.
%   coefficients = mfccCustom(x, fs) returns MFCCs for time-domain audio x.
%
%   Uses only basic arithmetic (suitable for later conversion to integer math).
%   Kept to single-channel, time-domain input for simplicity.
%
%   Options (passed as struct 'opts' with fields):
%     .Window          - Analysis window vector (default: periodic Hamming, 30ms)
%     .OverlapLength   - Overlap in samples (default: round(0.02*fs))
%     .FFTLength       - FFT length (default: window length)
%     .NumCoeffs       - Number of cepstral coefficients (default: 13)
%     .BandEdges       - Mel filter bank band edges in Hz (default: Slaney 42 edges)
%
%   Example:
%     [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
%     c1 = mfcc(audioIn,fs);
%     c2 = mfccCustom(audioIn,fs);
%     max(abs(c1(:)-c2(:)))   % should be ~0

    % =====================================================================
    % 1. Parse options (set defaults)
    % =====================================================================
    if nargin < 3, opts = struct(); end

    N=round(0.03*fs);
    win = zeros(1, N);
    for n = 0:N-1
        win(n+1) = 0.54 - 0.46 * cos(2*pi*n / N);
    end
    win = quantize(win,Q(1));
    winLength = round(0.03*fs);

    if isfield(opts,'FFTLength') && ~isempty(opts.FFTLength)
        fftLen = opts.FFTLength;
    else
        fftLen = winLength;
    end

    if isfield(opts,'NumCoeffs') && ~isempty(opts.NumCoeffs)
        numCoeffs = opts.NumCoeffs;
    else
        numCoeffs = 13;
    end

    factor = 400/3;
    bandEdges = zeros(1, 42);
    for ii = 1:13
        bandEdges(ii) = factor + (factor/2)*(ii - 1);
    end
    for ii = 14:42
        bandEdges(ii) = bandEdges(ii-1) * 1.0711703;
    end

    DHTmat = real(dftmtx(fftLen)) - imag(dftmtx(fftLen));
    DHTmat = quantize(DHTmat,Q(2));
    DHTmat = DHTmat';

    nyquist = fs / 2;

    % Clip band edges to Nyquist
    numValidEdges = 0;
    for i = 1:length(bandEdges)
        if bandEdges(i) <= nyquist + sqrt(eps)
            numValidEdges = numValidEdges + 1;
        end
    end
    if numValidEdges < length(bandEdges)
        edges = bandEdges(1:numValidEdges+1);
        edges(end) = nyquist;
    else
        edges = bandEdges;
    end

    numEdges = length(edges);
    numBands = numEdges - 2;

    % Determine valid edges (those <= fs/2)
    validNumEdges = 0;
    for i = 1:numEdges
        if (edges(i) - nyquist) < sqrt(eps)
            validNumEdges = validNumEdges + 1;
        end
    end
    validNumBands = validNumEdges - 2;

    % Linear frequency axis (Hz) for each FFT bin
    linFq = zeros(1, fftLen);
    for i = 0:fftLen-1
        linFq(i+1) = i / fftLen * fs;
    end

    % Find inflection points: for each edge, find the first FFT bin
    % whose frequency exceeds the edge.
    p = zeros(validNumEdges, 1);
    for edgeIdx = 1:validNumEdges
        for binIdx = 1:fftLen
            if linFq(binIdx) > edges(edgeIdx)
                p(edgeIdx) = binIdx;
                break;
            end
        end
    end

    % For the 'linear' FilterBankDesignDomain the triangles are drawn
    % directly in Hz (no mel/bark warping), so bandEdgesMod = edges,
    % FqMod = linFq.  (This is the default MATLAB mfcc behavior.)
    bw = zeros(1, numEdges-1);
    for i = 1:numEdges-1
        bw(i) = edges(i+1) - edges(i);
    end

    % Build the filter bank matrix (fftLen x numBands)
    filterBank = zeros(fftLen, numBands);
    for k = 1:validNumBands
        % Rising side of triangle
        if p(k) <= p(k+1)-1
            for j = p(k):p(k+1)-1
                filterBank(j, k) = (linFq(j) - edges(k)) / bw(k);
            end
        end
        % Falling side of triangle
        if p(k+1) <= p(k+2)-1
            for j = p(k+1):p(k+2)-1
                filterBank(j, k) = (edges(k+2) - linFq(j)) / bw(k+1);
            end
        end
    end

    % Bandwidth normalization: weight each band by (edge(k+2)-edge(k))/2
    for k = 1:numBands
        halfBW = (edges(k+2) - edges(k)) / 2;
        if halfBW ~= 0
            for j = 1:fftLen
                filterBank(j, k) = filterBank(j, k) / halfBW;
            end
        end
    end
    filterBank = quantize(filterBank/2,Q(3)); %embedded postscaling

    DCTmatrix=dctmtx(numBands)*log10(2); %embedded postscaling
    DCTmatrix=DCTmatrix(1:numCoeffs,:);
    DCTmat=quantize(DCTmatrix,Q(4))';
end

function x = quantize(x,Q)
    M=max(x,[],'all');
    x=x.*(2^Q/M);
    x=floor(x);
    x=x./(2^Q/M);
end