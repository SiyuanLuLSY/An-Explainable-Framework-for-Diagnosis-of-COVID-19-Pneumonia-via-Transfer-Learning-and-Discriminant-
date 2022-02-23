% load nets;


net = res;
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize(1:2);
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
img1 = imread("tb.jpg");%c n
dlnet = dlnetwork(lgraph);
img1 = imresize(img1,inputSize);
% img=img1;
img(:,:,1)=img1;
img(:,:,2)=img1;
img(:,:,3)=img1;
[classfn,score] = classify(net,img);
figure,imshow(img);
imhist(img)
% title(sprintf("%s (%.2f)", classfn, score(classfn)));
% type gradcam.m
softmaxName = 'softmax_out';
featureLayerName = 'res5b'; % 18: res5b 50: activation_49_relu
dlImg = dlarray(single(img),'SSC');
[featureMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, featureLayerName, classfn);
gradcamMap = sum(featureMap .* sum(dScoresdMap, [1 2]), 3);
gradcamMap = extractdata(gradcamMap);
% gradcamMap = rescale(gradcamMap,mn,mx);
gradcamMap = imresize(gradcamMap, inputSize, 'Method', 'bicubic');
imshow(img);
co=jet();
hold on;
imagesc(gradcamMap,'AlphaData',0.5);
colormap jet
% colorbar
% caxis([mn,mx])
hold off;
% title("Grad-CAM");