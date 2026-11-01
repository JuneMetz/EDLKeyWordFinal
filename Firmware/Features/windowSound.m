function frames = windowSound(x, fs)
    winLength = round(0.03*fs);
    overlapLength = round(0.02*fs);
    hopLength = winLength - overlapLength;
    numSamples = size(x, 1);
    numFrames = floor((numSamples - winLength) / hopLength) + 1;
    frames = zeros(numFrames, winLength);
    for f = 1:numFrames
        startIdx = (f-1)*hopLength + 1;
        frames(f, :) = x(startIdx:startIdx+winLength-1);
    end
end