function saveOutput

Folder = '/path/dataset/Output/';
File   = '01.png';
Img    = imread('/path/fatia0.png');
imwrite(Img, fullfile(Folder, File));



end