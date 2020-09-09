function [allp_sample, pose] = render_multicam( instance, mapper )

% It's not efficient to load the depth map every single time we want to
% visualize the data. The following code tries to take care of that.
if nargin < 2,
    if evalin('base', 'exist(''mapper'', ''var'')')
        mapper = evalin('base', 'mapper');
    else
        warning('Loading the default mapper');
        map_file = load('mapper.mat');
        mapper = map_file.mapper;
        assignin('base', 'mapper', mapper);
    end
end

% make sure we're dealing with the right data.
assert(numel(mapper)==1 || numel(mapper)==512*424, 'Bad input mapper');
assert(numel(instance)==1, 'One input at a time');

% Let's generate the cloud for our multicam instance.
clouds = generate_cloud_instance(instance, mapper);

% And now the groundtruth pose.
pose = get_pose(instance);


allp = cell2mat({clouds.cloud}');
sampleIndex = randperm(size(allp,1));
allp_sample = allp(sampleIndex(1:8192),:);

end
