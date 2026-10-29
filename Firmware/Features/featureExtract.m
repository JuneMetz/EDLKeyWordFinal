[x,fs] = audioread("XC820405.wav");

% Inputs required: x (audio signal), fs (sampling frequency)

% 1. Frame and window the signal
winLength = round(0.03*fs);
overlapLength = round(0.02*fs);
hopLength = winLength - overlapLength;
w = hamming(winLength,"periodic");
x = x(:);
numFrames = max(1,ceil((numel(x)-overlapLength)/hopLength));
x = [x; zeros(max(0,(numFrames-1)*hopLength+winLength-numel(x)),1)];
frames = buffer(x,winLength,overlapLength,"nodelay");
frames = frames(:,1:numFrames).*w;

% 2. Compute the magnitude spectrum
fftLength = numel(w);
spectrum = abs(fft(frames,fftLength,1));
spectrumLength = floor(fftLength/2)+1;
spectrum = spectrum(1:spectrumLength,:);

% 3. Construct the 40-filter mel filter bank
numFilters = 40;
mel = @(f) 2595*log10(1+f/700);
hz = @(m) 700*(10.^(m/2595)-1);
bandEdges = hz(linspace(mel(0),mel(fs/2),numFilters+2));
bin = floor((fftLength+1)*bandEdges/fs)+1;
filterBank = zeros(numFilters,spectrumLength);
for k = 1:numFilters
    l = bin(k); c = bin(k+1); r = bin(k+2);
    filterBank(k,l:min(c-1,spectrumLength)) = ((l:min(c-1,spectrumLength))-l)/(c-l);
    filterBank(k,c:min(r-1,spectrumLength)) = (r-(c:min(r-1,spectrumLength)))/(r-c);
end

% 4. Apply the filter bank
filterEnergies = filterBank*spectrum.^2;
filterEnergies = max(filterEnergies,realmin);

% 5. Apply logarithmic rectification
logFilterEnergies = log(filterEnergies);

% 6. Apply the DCT and retain 13 coefficients
coefficients = dct(logFilterEnergies);
coefficients = coefficients(1:13,:).';

% 7. Append log frame energy
logEnergy = log(max(sum(frames.^2,1),realmin)).';
mfccCoefficients = [coefficients logEnergy];

plot(mfccCoefficients)