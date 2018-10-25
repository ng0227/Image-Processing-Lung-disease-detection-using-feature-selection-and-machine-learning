function readInput

srcFiles = dir('/path/Grayscale/Healthy/*.png');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('/path/dataset/Grayscale/Healthy/',srcFiles(i).name);
    I = imread(filename);
    figure, imshow(I);
end

end