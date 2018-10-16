function lungSegment

%code for segmentation using ground truth


imageDir = dir('/path/dataset/Grayscale/COPD/*.png');  % the folder in which ur images exists
maskDir = dir('/path/dataset/GroundTruth/COPD/*.png');
outputDir = '/path/dataset/Output/COPD';


for i = 1 : length(imageDir)
    imageFilename = strcat('/path/dataset/Grayscale/COPD/',imageDir(i).name);
    maskFileName = strcat('/path/dataset/GroundTruth/COPD/',maskDir(i).name);
    
    
    image = imread(imageFilename);
    
    maskImage=imread(maskFileName);
    maskImage=rgb2gray(maskImage);
    
    mask = zeros(size(maskImage));
    mask(25:end-25,25:end-25) = 1;
    
    bw = activecontour(maskImage,mask,2000);
    binary = logical(bw);
    
    image(~binary)=0;
    outputFileName = sprintf('%d.png',i);
    imwrite(image, fullfile(outputDir, outputFileName));
     
end

disp('finish');



end
