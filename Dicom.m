function Dicom

filename = '/path/dicomImage1.dcm';
X = dicomread(filename);

[Y,map]=dicomread(filename);

info = dicominfo(filename);
imshow(X,'DisplayRange',[]);
end