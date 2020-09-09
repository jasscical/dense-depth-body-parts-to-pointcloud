addpath(genpath('./models/'));
addpath(genpath('./ubc3vtoolkit'));

% Update this if you already have a matconvnet on your computer.
% It requires compilation.
run ./matconvnet/matlab/vl_setupnn.m

%% Load and prepare the image.
png_name = 'mayaProject.000983.png';
png_path = strcat('./5_data/', png_name);
save_path = strcat('./100_data/pre_250_250/' , png_name);

depth_image = importdata(png_path); %��ȡpng_pathΪ����depth_image

% Find the boundaries of the person in the depth image.
margin_1 = sum(depth_image.alpha, 1); %depth_image.alpha�ĵ� 1 ά�ȣ���������ֵ����ӣ���Ϊ�������ұ߽磩margin_1
margin_2 = sum(depth_image.alpha, 2); %depth_image.alpha�ĵ� 2 ά�ȣ���������ֵ�� ��ӣ���Ϊ�������±߽磩margin_2
%���±߽�margin_2
top = find(margin_2, 1, 'first');%Ѱ��margin_2ǰ1����0Ԫ��,�ϱ߽�
bottom = find(margin_2, 1, 'last');%Ѱ��margin_2������1����0Ԫ�أ��±߽�
%���ұ߽�margin_1
left = find(margin_1, 1, 'first');%Ѱ��margin_1ǰ1����0Ԫ�أ���߽�
right = find(margin_1, 1, 'last');%�ұ߽�

assert( top<bottom && left<right, 'Top/Bottom & Left/Right are not consistent');
%����Ŀ���
width = right-left+1;
height = bottom-top+1;


% Crop the image
depth_image.cdata = depth_image.cdata(top:bottom, left:right, 1);
depth_image.alpha = depth_image.alpha(top:bottom, left:right);

target_window = 190;
margin = 30;
%���ű���scale
if height > width,
    scaler = [target_window NaN]; %��� ��>��height������target_window,width��С��������target_window��
else                              %Ϊ��heightһ��������width������ NaN ���棿��
    scaler = [NaN target_window];    
end

% Scale the image
depth_im = imresize(depth_image.cdata, scaler, 'lanczos2'); %����depth_image.cdata��scaler�����ı�ͼ��ߴ�ʱ����lanczos2�㷨��depth_im�����ƴ���䱳���飿
depth_mask = imresize(depth_image.alpha, scaler, 'nearest') == 255; %���ı�ͼ��ߴ�ʱ��������ڲ�ֵ�㷨��depth_mask�Ǵ洢��alphaֵ���������alphaֵ����255�ĲŸ�ֵ��depth_mask

average_depth = mean(depth_im(depth_mask));%depth_im �е� depth_maskƽ�����
depth_im(depth_mask) = uint8(  int64(depth_im(depth_mask)) + int64(55-average_depth) ); %��depth_im �е� depth_mask ���в���
depth_im(~depth_mask) = 255; %��depth_im �е� �� depth_mask ȫ����ֵΪ 255

depth_im = single(depth_im)/255;
[height, width] = size(depth_im);

final_depth_image = ones(target_window + 2*margin, target_window + 2*margin, 'single'); %��һ��190+2*30 �� 190+2*30��
final_depth_mask = zeros([target_window + 2*margin, target_window + 2*margin], 'uint8');
loc_x = floor((target_window + 2*margin - width)/2)+1;
loc_y = floor((target_window + 2*margin - height)/2)+1;

final_depth_image(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_im;
final_depth_mask(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_mask*255;

%% Load and prepare the network.
net = load('hardpose_69k_dag');
net = dagnn.DagNN.loadobj(net);

net.eval({'data', final_depth_image});

prediction = net.getVar('prob').value;
%% Visualize the predcitions

[~, classes] = max(prediction, [], 3);%����Prediction�е�3ά�������ά�ȸ�classes
classes(~final_depth_mask) = 0;
imagesc(classes);

MappedData = mapminmax(classes, 0, 1);%���ݹ�һ��

% filename = 'G:/dense-depth-body-parts-master/classes.txt';
% dlmwrite(filename,classes);


%% Save the classes image
imwrite(MappedData,save_path,'png');
fprintf('end----------------');



% set(gca,'XTick',[]) % Remove the ticks in the x axis!
% set(gca,'YTick',[]) % Remove the ticks in the y axis
% set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
% saveas(gcf,png_path,'png')



